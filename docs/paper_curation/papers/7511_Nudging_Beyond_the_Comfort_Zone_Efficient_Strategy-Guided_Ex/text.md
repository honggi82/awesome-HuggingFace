# arXiv:2605.15726v1[cs.AI]15May2026

## Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR

Chanuk Lee1* Sangwoo Park1* Minki Kang1 Sung Ju Hwang1,2 1KAIST 2DeepAuto.ai {tallyforce, swgger}@kaist.ac.kr

### Abstract

Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While increasing the number of rollouts alleviates this issue, such brute-force scaling is computationally expensive, and existing approaches that modify the optimization objective provide limited control over what is explored. In this work, we propose NUDGERL, a framework for structured and diversity-driven exploration in RLVR. Our approach introduces Strategy Nudging, which conditions each rollout on lightweight, strategy-level contexts to induce diverse reasoning trajectories without relying on expensive oracle supervision. To effectively learn from such structured exploration, we further propose a unified objective, which decomposes the reward signal into interand intra-context components and incorporates a distillation objective to transfer discovered behaviors back to the base policy. Empirically, NUDGERL outperforms standard GRPO with up to 8× larger rollout budgets, while outperforming oracle-guided RL baseline on average across five challenging math benchmarks. These results demonstrate that structured, context-driven exploration can serve

- as an efficient and scalable alternative to both brute-force rollout scaling and feasibility-oriented methods based on privileged information. Our code is available
- at https://github.com/tally0818/NudgeRL.

### 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for improving the reasoning capabilities of large language models (LLMs) [7, 20]. By leveraging verifiable rewards, methods such as Group-Relative Policy Optimization (GRPO) [18] enable scalable post-training without requiring dense supervision. This paradigm has been successfully applied across a wide range of domains.

Despite its success, RLVR remains fundamentally limited by its ability to explore the space of reasoning trajectories. A natural approach is to scale the number of sampled rollouts, which increases the probability of discovering rare trajectories [5]. However, such brute-force scaling quickly becomes computationally prohibitive, motivating alternative approaches that improve exploration efficiency.

Recent work has sought to address this limitation by modifying the optimization objective, for example through entropy regularization or decoupled clipping [24, 26]. While these methods encourage broader exploration at the distribution level, they provide limited control over what is explored, and often fail to ensure coverage of semantically meaningful reasoning strategies. Another

*Equal contribution

Preprint.

- Rollout 1 A

- Rollout 2 A

- Rollout 3 B

###### Naive Sampling collapses to the Dominant Mode

6        1      

###### Question

𝜋 LLM

1     6  S    

[Figure 1]

Determine all functions 𝑓:ℕ → ℝ satisfying 𝑓 𝑥 + 𝑦 + 𝑓 𝑥 − 𝑦 = 𝑓(3𝑥) for all 𝑥,𝑦

   F                

(a) Naive Sampling (GRPO)

- Rollout 1 A

- Rollout 2 B

###### Standard Nudging allows sampling Diverse Modes

- Q + A

- Q + B

- Q + C

𝜋

LLM Rollout 3 C

(b) Strategy Nudging (NudgeRL)

[Figure 2]

      F    

B: Linear functions and their properties

C: Induction on natural numbers

Potential Strategies

A: Cauchy’s functional equation

(c) Effect of Strategic Nudging

- Figure 1: Concept: Improving exploration diversity through Strategy Nudging. (a) Naive sampling methods (e.g., GRPO) often collapse to a dominant reasoning mode, limiting the exploration of the reasoning space. (b) NUDGERL introduces Strategy Nudging, which appends lightweight strategy to the input, forcing the model to traverse diverse reasoning modes. (c) As a result, Strategy Nudging significantly increases the number of distinct reasoning approaches discovered compared to the baseline, effectively mitigating the exploration bottleneck. Additional details are in Sec. B

line of work leverages privileged information, such as oracle solutions or intermediate reasoning steps, to improve the feasibility of discovering correct trajectories [8, 16, 19, 27]. Although effective, these approaches are primarily feasibility-oriented and rely on strong supervision signals that are expensive to obtain and difficult to scale. Moreover, by guiding the policy toward a narrow set of predefined successful trajectories, they may limit exploration diversity and hinder the discovery of alternative reasoning strategies [23, 25].

In this work, we address the exploration bottleneck by explicitly structuring the reasoning space in a scalable manner. We propose NUDGERL, a framework that introduces Strategy Nudging during the exploration phase. Instead of relying on expensive oracle data, Strategy Nudging appends lightweight, heuristic text prompts (e.g., specific strategies for math problems or reasoning keywords) to the input. This deliberately forces the model to traverse distinct, diverse reasoning modes that it might otherwise ignore under purely naive sampling.

However, learning from such context-conditioned exploration introduces new challenges. Since rollouts are generated under different context-conditioned prompts, the samples are naturally partitioned into multiple distinct groups, where reward variation reflects both the intrinsic trajectory quality and context-specific biases, making standard group-wise advantage estimation unreliable. Furthermore, context forcing creates a mismatch between how trajectories are sampled and how the policy is finally used at inference time. Without intervention, improvements discovered under context-forced exploration may not transfer directly to the base policy. To address these challenges, we further introduce (i) an Inter-Intra group advantage to enable meaningful credit assignment across context-induced groups, and (ii) a distillation-augmented objective that explicitly transfers effective behaviors discovered during context-forced exploration back to the base policy.

Our approach enables structured and diversity-driven exploration while remaining fully compatible with standard RLVR pipelines. Empirically, NUDGERL achieves performance surpassing GRPO even when GRPO is given an 8× larger rollout budget, while outperforming oracle-guided baselines. This demonstrates that scalable, diversity-oriented exploration can serve as an effective alternative to both brute-force rollout scaling and feasibility-driven privileged information.

### 2 Preliminaries

#### 2.1 Group-Relative Policy Optimization (GRPO)

We consider an empirical distribution of prompts x0 ∈ D. For each prompt x0, a policy πθ generates a group of N rollouts {yi}Ni=1, where each rollout is sampled as yi ∼ πθ(· | x0). Each rollout is evaluated by a verifiable reward function R(x0,yi) ∈ {0,1}.

Unlike standard PPO [17], which typically estimates advantages using a learned value function, GRPO [18] derives advantages from group-wise rewards. For rollouts sampled from the same prompt x0, let ri = R(x0,yi) denote the reward of rollout yi. The group-wise advantage is then defined as:

ri − µ σ + δ

Aˆi =

, (1)

where µ and σ are the reward mean and standard deviation within the group, and δ > 0 is used for numerical stability. This yields a relative advantage estimate without training a value function.

The policy is then optimized with a PPO-style clipped objective:

πθ(y|x0) πold(y|x0)

θ(·|x0) min r0A,ˆ clip(r0,1 − ϵ,1 + ϵ)Aˆ , where r0 =

. (2)

LGRPO(θ) = −Ey∼π

Thus, GRPO retains PPO’s clipped objective while using group-relative advantages.

#### 2.2 Motivation: From Exploration to Performance Gain

To understand why exploration is a fundamental bottleneck in RLVR, we look beyond trajectory-level rewards and examine how the probability mass of generated tokens shifts during training. Hu et al. [5] characterizes the expected one-step performance improvement (∆Qpos) in RLVR as:

η N

(1 − SR)QnegA2 + SRQposB2 + SR(QposUneg,2 − QnegUpos,2) , (3)

∆Qpos =

where Qpos and Qneg = 1 − Qpos denote the total probability mass of correct and incorrect tokens, η is the learning rate, and N is the number of rollouts. A2 and B2 are the second moments of sampled correct and incorrect tokens, while Upos,2 and Uneg,2 are those of unsampled correct and incorrect tokens. SR ∈ [0,1] represents the net reward contribution from sampled tokens.

Since SR ∈ [0,1], the first two terms in Eq. 3 are non-negative and drive learning forward. The third term, however, acts as a potential penalty. Because incorrect tokens typically dominate the probability

mass (Qneg ≫ Qpos), a large Upos,2, meaning the model has significant probability mass on correct trajectories that it simply fails to explore, creates a dominant negative force that hinders performance gain. Therefore, the core bottleneck of RLVR lies in the unexplored correct regions.

Limitations of rollout scaling. To mitigate this penalty, a naive solution is to increase the rollout size N. Hu et al. [5] shows that for a collection of tokens with probabilities {pi}, the expected unsampled second moment after N draws is:

i

p2i(1 − pi)N, (4)

which decreases monotonically with N. However, tokens with small pi decay slowly, so fully covering long-tail correct trajectories requires prohibitively large rollout budgets.

This highlights the limitation of blindly scaling N to reduce the unexplored correct mass (Upos,2). Long-tail correct trajectories remain unlikely to be sampled even under large N, suggesting the need for a structured exploration mechanism that can efficiently expose such latent trajectories.

### 3 NUDGERL

We introduce NUDGERL, a framework for structured exploration and learning in RLVR. NUDGERL consists of three components: (i) Strategy Nudging, which conditions rollout generation on strategylevel contexts to induce diverse reasoning trajectories; and (ii) Inter-intra Group Advantage, a credit assignment method that enables controlled exploration and exploitation of strategies; and (iii) Distillation augmented RL objective to learn from context-conditioned rollouts and distill effective strategies into the policy under the original prompt for inference without external context.

#### 3.1 Strategy Nudging: Structured Exploration via Strategy-Level Contexts

Given that prior work [5] alleviates the exploration bottleneck by reducing unsampled probability mass through larger rollout budgets, a natural question arises: how many rollouts are required to reliably discover a rare trajectory? To quantify this discovery cost, consider a rare trajectory y with π(y|x0) ≪ 1. The expected number of rollouts required to observe y at least once is:

1 π(y|x0)

E # rollouts =

. (5)

Rollout 1

Rollout 2

Rollout 3

| | |
|---|---|
| | |

- Q + A

- Q + B

|𝜋𝜃|
|---|

Q + B Rollout

|𝜋𝜃|
|---|

| | |
|---|---|
| | |

𝑥1

| | |
|---|---|
| | |

| | |
|---|---|
| | |

ℒDistill

LLM

LLM Rollout 4

Rollout 5

Rollout 6

Q

/ / Positive/NegativeAdvantage

|𝜋𝜃|
|---|

[Figure 3]

[Figure 4]

| | |
|---|---|
| | |

Correct / Wrong Reward

𝑥0

(a) Inter-Intra Group Advantage

(b) Self-distillation

- Figure 2: Overview of the NudgeRL learning mechanism. (a) Inter-Intra Group Advantage: Demonstrates credit assignment that emphasizes reliable contexts (i.e., λ ∈ (1,2]). A successful rollout from a consistently high-reward context (Strategy B) receives a larger positive advantage than a rare success from a low-reward context (Strategy A). (b) Self-distillation: Illustrates bridging the train-test gap. High-quality trajectories discovered via context-conditioned exploration (Q + B)

are distilled back into the base policy (Q) using LDistill, allowing the model to internalize effective reasoning modes for context-free inference.

This implies that for low-probability trajectories, the required rollout budget grows prohibitively large. In practice, naive rollout scaling repeatedly samples from high-probability modes of the current policy, leading to diminishing returns in covering rare trajectories.

This motivates conditioning generation on a context c that can shift the sampling distribution toward otherwise rare trajectories. If such a context increases the probability of a trajectory y, i.e., π(y|x,c) ≫ π(y|x), then its expected number of rollouts becomes:

1 π(y|x0)

1 π(y|x0,c) ≪

. (6)

E # rollouts | c =

Thus, contexts need not provide a solution; they can serve as lightweight controls that alter the sampling distribution and reduce the cost of discovering rare trajectories.

Strategy Nudging. Even though context conditioning can improve exploration efficiency in principle, simply placing multiple contexts in a single prompt leaves the choice of strategy to the policy, which may ignore some contexts and repeatedly follow dominant reasoning patterns. To enforce coverage over contexts, we instead assign a single sampled context to each rollout before generation.

Let C(x0) = {c1,...,cM} denote a pool of Strategy-level contexts for the original prompt x0. For each rollout index i, we begin with sampling c(i) ∼ Uniform(C(x0)). To avoid relying exclusively on the context pool and to retain compatibility with the original prompt, we further apply context dropout. Specifically, we sample a mask b(i) ∼ Bernoulli(1 − pdrop) and define the context as:

c(i), b(i) = 1, ∅, b(i) = 0.

z(i) =

(7)

We then construct the final prompt x(1i) = (x0,z(i)), and generate yi ∼ πθ(· | x(1i)). By varying z(i) across rollout indices, Strategy Nudging induces diversity at the input-conditioning level, rather than

relying solely on sampling from a single prompt. Details on generating C are in Sec. B.

Context-induced rollout diversity. To verify that Strategy Nudging induces the intended diversity, we compare it against naive sampling without context conditioning. For each prompt, both methods generate 8 rollouts in total: Strategy Nudging samples 4 rollouts from each of 2 contexts without context dropout, whereas the baseline samples all 8 rollouts from the base policy under the original prompt. We then cluster the reasoning structures using an LLM-as-a-judge (gpt-4o-mini [15]) and measure the number of distinct clusters; additional details are provided in Sec. B.

As shown in Fig. 1, Strategy Nudging more often increases the number of distinct reasoning structures relative to naive sampling, whereas the base policy frequently collapses to similar patterns. This suggests that Strategy Nudging diversifies exploration before any policy update is applied, allowing the rollout set to cover a broader range of reasoning modes under the same rollout budget.

#### 3.2 Inter-Intra Group Advantage: Learning to Balance Exploration between Strategies

GRPO estimates advantages by comparing rewards among rollouts conditioned on the same prompt distribution. With Strategy Nudging, however, rollouts are drawn from context-conditioned prompts

(x0,z(i)). A single group baseline therefore entangles reward variation induced by different contexts, distorting the relative advantage assigned to each rollout.

To address this, we propose the Inter-Intra Group Advantage, which assigns credit through two complementary signals: an intra-context signal, capturing trajectory quality under the same conditioning context, and an inter-context signal, capturing the relative reliability of the context itself.

Given sampled rollouts {yi}Ni=1 with rewards ri = R(x0,yi), we group them according to their assigned contexts. The set of context groups is defined as

G(x0) = Unique({z(i)}) ⊆ C(x0) ∪ {∅}. (8)

For each group g ∈ G(x0), we define the index set Ig = {i | z(i) = g}, which partitions all rollouts. We then compute both context-level and global reward baselines:

1 |Ig| i∈I

r¯g =

ri, r¯ =

g

Using these baselines, we define the advantage as:

N

1 N

ri. (9)

i=1

(ri − r¯z(i)) + λ(¯rz(i) − r¯) if z(i) ̸= ∅, ri − r¯ if z(i) = ∅.

Ai − µA σA + δ

Aˆi =

, where Ai =

(10)

µA and σA are the mean and standard deviation of {Ai}, and δ > 0 ensures numerical stability.

Because advantages determine direction of the policy update, they should remain consistent with the underlying rewards while allowing context-level preferences to affect credit assignment.

Proposition 3.1. Consider two trajectories y and y′ sampled from context groups z and z′, with rewards r and r′, respectively. Let r¯z and r¯z′ denote the corresponding context means, and let A and A′ denote their advantages. In the binary reward setting, if λ ∈ [0,2], then:

r > r′ ⇒ A > A′. (11)

Thus, for λ ∈ [0,2], a higher reward always receives a higher advantage, ensuring consistency with the underlying objective; context only affects the relative ordering among equal-reward trajectories. For equal-reward trajectories, λ controls the context-level preference: λ < 1 favors successes from lower-reward contexts, encouraging exploration of less typical contexts, whereas λ > 1 favors successes from higher-reward contexts, emphasizing more reliable contexts. The neutral case λ = 1 treats equal-reward trajectories identically across contexts; the λ > 1 case is illustrated in Fig. 2 (a).

#### 3.3 Training objective

Although Strategy Nudging improves exploration by sampling rollouts from context-conditioned prompts x1 = (x0,z), the target policy at inference time should operate without external contexts. Therefore, useful trajectories discovered under x1 must be transferred to the base policy πθ(· | x0).

To bridge this gap, we introduce an advantage-weighted distillation term following Song et al. [19], which directly updates the policy using trajectories sampled under the context-conditioned input x1:

θ(·|x1) A ˆlog πθ(y|x0) . (12)

LDistill(θ) = −Ey∼π

Unlike standard behavior cloning, this formulation selectively emphasizes trajectories with high normalized advantage, ensuring that only useful behaviors discovered under diverse contexts contribute to the update of πθ(· | x0).

In parallel, we optimize the reinforcement learning objective on the context-conditioned policy:

πθ(y|x1) πold(y|x1)

θ(·|x1) min(r1A,ˆ clip(r1,1 − ϵlow,1 + ϵhigh)Aˆ) , wherer1 =

. (13)

LRL(θ) = −Ey∼π

The final objective combines both terms:

LNUDGERL = LRL + λdistillLDistill. (14)

Table 1: Main results comparing rollout scaling, oracle hinting, and context-based exploration. We report pass@1 estimated using 128 rollouts. Best results are represented as bold and second best as underline. † indicates additional implementation details; see Sec. C for details.

Model Method Rollouts (N) AIME24 AIME25 AMC23 MATH500 APEX Average

Base model – 0.374 0.352 0.653 0.592 0.036 0.402

8 0.444 0.367 0.749 0.668 0.040 0.454 16 0.454 0.355 0.840 0.655 0.045 0.470 32 0.451 0.370 0.881 0.674 0.058 0.487 64 0.415 0.324 0.848 0.641 0.027 0.451

GRPO

Qwen3-4BInstruct

POPE† [16] 8 0.460 0.337 0.838 0.652 0.048 0.467 NUDGERL 8 0.482 0.393 0.857 0.660 0.053 0.489

Base model – 0.134 0.118 0.467 0.384 0.021 0.225

8 0.187 0.159 0.537 0.434 0.025 0.268 16 0.188 0.176 0.548 0.461 0.023 0.279 32 0.195 0.176 0.553 0.459 0.024 0.281 64 0.081 0.053 0.349 0.291 0.027 0.160

GRPO

Olmo3-7BInstruct-SFT

POPE† [16] 8 0.186 0.169 0.558 0.460 0.023 0.279 NUDGERL 8 0.190 0.179 0.563 0.468 0.025 0.285

This objective induces a complementary learning dynamic. The RL term operates on the contextconditioned policy, improving exploration and reinforcing successful trajectories within each context. In contrast, the distillation term projects these improvements onto the base-prompt policy, enabling cross-context generalization. As a result, the model learns to reproduce effective reasoning strategies without relying on explicit context at inference time. Unlike GRPO in Eq. 2, which samples and optimizes trajectories under the original prompt x0, NUDGERL performs RL on context-conditioned rollouts under x1 while distilling high-advantage trajectories back into the base policy πθ(·|x0).

### 4 Experiments

#### 4.1 Experimental Setup

Baselines. We compare our method against (i) the base model without optimization, which serves as the reference point; (ii) GRPO with increasing rollout budgets, where N ∈ {8,16,32,64}, which evaluates naive rollout scaling as a brute-force exploration strategy; and (iii) POPE [16], which augments standard GRPO by appending prefixes of the oracle solution at the end of the base prompt, thereby alleviating the sparse reward signal bottleneck. Further details are provided in Sec. C.

Evaluation Datasets and Metrics. AIME24 and AIME25, 30-problem olympiad-style high-school competitions [13]; AMC23, a 40-problem high-school contest benchmark [12]; the level-5 subset of MATH500, containing 134 difficult MATH problems [4]; and the Apex Shortlist, consisting of 48 advanced competition-style problems [1]. We report pass@1, estimated from 128 rollouts using the unbiased estimator of Chen et al. [2]. All solutions are automatically graded using math-verify [6]. Additional details are provided in Sec. E.

Implementation Details. We apply NUDGERL to Qwen3-4B-Instruct-2507 [21] and Olmo-3-7B-Instruct-SFT [14] using DAPO-17k-Processed as a training set [24]. To construct the pool of contexts, we used gpt-4o-mini [15] to generate two strategy-level contexts per problem (e.g., Pythagorean theorem), and used them without additional verification (i.e., ∀x ∈ D,|C(x)| = 2). For the POPE baseline, oracle solutions were generated using DeepSeek Reasoner v3.2 [9]. We provide additional optimization details in Sec. D.

#### 4.2 Main Results

NudgeRL matches larger-budget GRPO with fewer rollouts. As shown in Tab. 1, NUDGERL achieves the best average performance on both models while using only 8 rollouts per prompt. On Qwen3-4B-Instruct-2507, NUDGERL reaches 0.489 average pass@1, slightly outperforming the best GRPO result at 32 rollouts (0.487) and surpassing GRPO at 64 rollouts (0.451) with an 8×

0.9

Base model

| |
|---|

GRPO-8

0.8

0.42

0.6

GRPO-16 NudgeRL

Averagepass@1

| |
|---|

0.7

pass@k

Reward

| |
|---|

0.40

0.5

| |
|---|

0.6

GRPO-8

0.38

0.4

0.5

GRPO-8

GRPO-16 GRPO-32 GRPO-64 NudgeRL

GRPO-16 GRPO-32 NudgeRL

0.4

0.36

0.3

1 4 16 64 256 1024

100 200 300 400 500

100 200 300 400 500

k

Step

Step

(a) Training reward

(b) AIME24/25 pass@1

(c) AIME24/25 pass@k

- Figure 3: Training dynamics and evaluation performance on Qwen3-4B-Instruct. (a) EMAsmoothed training reward with decay factor 0.99. (b, c) Average pass@1 and pass@k on AIME24/25, estimated from 64 sampled rollouts using the unbiased estimator.

smaller rollout budget. On Olmo3-7B-Instruct-SFT, NUDGERL likewise improves over the best GRPO result, achieving 0.285 compared to 0.281 at 32 rollouts. These results indicate that larger rollout budgets alone are not sufficient: GRPO improves up to N = 32 but degrades at N = 64 on both models, suggesting instability under brute-force rollout scaling. In contrast, NUDGERL achieves stronger performance by improving the quality of exploration through Strategy Nudging, rather than relying on more sampled rollouts.

Comparison with oracle-prefix method. We also compare with POPE [16], which augments GRPO by generating rollouts conditioned on the oracle solution prefixes. Unlike baselines relying on expensive, unscalable oracle hints [16] or text feedback [19], our approach ensures scalable diversity. We use a lightweight LLM (e.g., gpt-4o-mini) to cheaply generate unverified strategy-level contexts that induce multiple reasoning directions. Despite this weaker supervision, our method consistently outperforms oracle-guided baselines, demonstrating that structured exploration over diverse strategies is more effective than injecting narrow, privileged solution signals.

#### 4.3 Efficient Coverage of Diverse Reasoning Modes

As discussed in Sec. 3.1, relying solely on scaling the rollout budget suffers from severe sample inefficiency when discovering long-tail, low-probability reasoning modes. This is because naive rollout scaling repeatedly allocates computation to dominant trajectories. To empirically investigate how Strategy Nudging overcomes this exploration bottleneck and improves sample efficiency, we compare the training dynamics of NUDGERL against GRPO under progressively larger rollout budgets. We evaluate the model for every 50 training steps on the combined AIME24 and AIME25 benchmark by sampling 64 rollouts per problem and estimating pass@1 and pass@8.

- As shown in Fig. 3b, NUDGERL improves pass@1 faster than GRPO variants and remains the strongest method throughout most of training. By 200 steps, NUDGERL exceeds 0.42 pass@1 on AIME24/25, while GRPO variants remain around or below 0.41 and show slower or less stable gains as the rollout budget increases. This suggests that Strategy Nudging improves sample efficiency by exposing useful reasoning trajectories earlier, rather than merely increasing sampled rollouts. Enlarging the number of samples (k) further validates this trend under the same training rollout budget. As shown in Fig. 3c, NUDGERL consistently outperforms GRPO-8 across the full k range, which indicates that Strategy Nudging improves inference-time sample efficiency, requiring fewer generated solutions to reach the same level of pass@k.

4.4 Case Study

To examine the source of performance gains in NUDGERL, we analyze one AIME25 problem where the NUDGERL-trained model successfully sampled correct trajectories, while the GRPO-trained model entirely failed. We sampled 32 rollouts and categorized their dominant reasoning strategies.

- As shown in Fig. 4, both models predominantly relied on coordinate geometry. However, the GRPOtrained model additionally explored ineffective strategies such as symmetry assumptions and area decomposition, which consistently resulted in truncated solutions, causing all 32 trajectories to fail. While GRPO sampled the shoelace formula strategy only once, NUDGERL substantially increased its frequency and successfully exploited it to generate correct trajectories.

Correct Shoelace 6/10

4

6

Strategy / outcome

coordinate geometry trigonometry symmetry assumption area decomposition

vector/equilateral geometry shoelace formula Truncated & wrong

13

| |
|---|

| |
|---|

32/32 Truncated & wrong

6/32 Correct

5

15

4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Correct Shoelace

| |
|---|

4

5

5

GRPO

NudgeRL

- Figure 4: NUDGERL internalizes effective test-time strategies. Across 32 rollouts on a AIME25 problem, GRPO yields only incorrect and truncated trajectories. Conversely, NUDGERL produces 6 correct solutions using the shoelace formula.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | |Hint<br><br>Drop|ed Reward out Reward| |

100 200 300 400 500

Step

0.3

0.4

0.5

0.6

ContextReward

- Figure 5: Training dynamics. We report time-weighted EMA reward mean (0.99) with and without context.

NudgeRL

NudgeRL

0.60

0.60

Average@1pass

Average@1pass

0.58

0.58

0.56

0.56

0.54

0.00 0.25 0.50 0.75 pdrop

Random Top ranked Hint sampling

(a) Ablation on pdrop

(b) Ablation on sampling

Figure 6: Ablation results on sampling. We report Average pass@1 estimated using 128 rollouts on AIME24/25, AMC23, MATH500.

This behavior highlights the complementary roles of our framework: Strategy Nudging exposes rare but effective reasoning modes such as the shoelace-formula strategy, while the Inter-Intra Group Advantage reinforces and exploits such reliable strategies once discovered. Details are in Sec. F.

#### 4.5 Effect of Contexts during training

We also report the dropout reward mean (Ei[r(x0,y(i)) | z(i) = ∅]) and the hinted reward mean (Ei[r(x0,y(i)) | z(i) ̸= ∅]) during training of Qwen3-4B-Instruct-2507 with NUDGERL. As shown in Fig. 5, both rewards improve together throughout training, suggesting that trajectories discovered under context-conditioned exploration are successfully transferred to the base policy through the distillation objective. Interestingly, the dropout reward occasionally exceeds the hinted reward during training. This contrasts with prior feasibility-oriented methods based on privileged information [8, 16, 19, 27]. In ours, primary role of context is not to directly simplify the problem, but to induce diverse reasoning trajectories that can later be internalized by the context-free policy.

#### 4.6 Underlying Mechanism of NUDGERL

To further understand the source of performance gains in NUDGERL, we conduct a series of controlled experiments using Qwen3-4B-Instruct-2507 [21] on a subset of benchmarks.

pdrop Ablation. As shown in Fig. 6a, a moderate dropout rate (pdrop = 0.5) consistently yields the best performance across benchmarks. Context dropout plays a dual role: it enables exploration beyond fixed contexts by occasionally reverting to the base prompt, while also stabilizing group-wise statistics through a more balanced sample distribution. When pdrop = 0, exploration is restricted to predefined contexts, whereas large values diminish the influence of context forcing. These results suggest that maintaining a balanced mixture of context-conditioned and context-free samples is important for achieving both diverse exploration and stable optimization.

Hint Sampling. We study how the quality of sampled contexts affects performance by comparing two strategies: random sampling and top-ranked selection. In the top-ranked setting, we first generate a pool of five candidate contexts, and then select the two that yield the largest improvement in pass@16 for each problem, as measured by oracle evaluation.

NudgeRL

NudgeRL

0.60

0.60

0.60

Average@1pass

Average@1pass

Average@1pass

0.58

0.58

0.58

0.56

0.56

0.56

NudgeRL ( high=0.20)

0.54

0.54

GRPO

0.0 0.1 0.5

0.20 0.24 0.28 0.32 0.36 0.40

0.9 1.0 1.1

high

distill

(a) Ablation results on λ

(b) Ablation results on λdistill

(c) ϵhigh scaling results

Figure 7: Ablation on learning and ϵhigh scaling results. We report Average pass@1 estimated using 128 rollouts on AIME24/25,AMC23,MATH500 dataset.

As shown in Fig. 6b, random sampling consistently outperforms top-ranked selection in terms of pass@1. While top-ranked contexts ensure more correctness, they tend to concentrate on a narrow set of reasoning strategies. In contrast, random sampling induces a broader distribution over plausible trajectories, resulting in more effective exploration under limited rollout budgets.

These results suggest that, within our framework, the primary role of context is not to provide the single best hint, but to promote diversity in reasoning. Consequently, simple random sampling is not only sufficient, but also preferable for scalable and effective context-based exploration.

Exploration-Exploitation trade-off via λ. Fig. 7a presents the effect of varying λ, where λ = 1.1 achieves the best performance. This trend aligns with our Proposition 3.1 in Sec. 3.2. Since strategy nudging already ensures sufficient diversity at the sampling stage, increasing λ does not hinder exploration across contexts. Instead, it strengthens exploitation within each problem by prioritizing trajectories from more reliable contexts. This leads to more consistent learning of high-quality solutions per instance, explaining the observed performance gains at λ = 1.1.

Distillation Coefficient. As shown in Fig. 7b, removing the distillation term (λdistill = 0) results in a clear performance drop, indicating that explicitly transferring context-discovered trajectories to the base policy is essential. However, overly large values also degrade performance, likely due to over-constraining the policy toward sampled trajectories. A moderate coefficient (λdistill = 0.1) achieves the best results, suggesting that distillation should complement the underlying RL objective.

#### 4.7 Comparison with ϵhigh scaling.

We further compare our algorithm with decoupled clipping [24]: clip(r,1 − ϵlow,1 + ϵhigh), where ϵhigh controls the strength of policy updates by amplifying the contribution of successful trajectories. Increasing ϵhigh therefore allows more aggressive policy updates toward positive-advantage trajectories. As shown in Fig. 7c, increasing ϵhigh generally improves GRPO performance in the moderate regime used in prior works [18, 24]. However, our method with ϵhigh = 0.2 consistently outperforms GRPO across the entire scaling range from moderate to extreme values. This suggests that improving exploration quality is more effective than simply increasing the magnitude of stochastic policy updates. Additionally, under the more extreme scaling adopted in recent RLVR settings [10], GRPO sharply deteriorates at ϵhigh = 0.4. We argue that this degradation highlights a limitation of purely stochastic distribution-level exploration: increasing update magnitude alone provides little control over what is explored.

The complete results of the evaluation are given in the Sec. G.

### 5 Conclusion

In this work, we introduced NUDGERL, a framework for structured exploration in RLVR. Our approach leverages Strategy Nudging to induce diverse reasoning trajectories by sampling from lightweight, strategy-level context-conditioned distributions, and learns from them via distillation augmented RL objective. Empirically, NUDGERL achieves superior performance compared to GRPO using up to 8× larger rollout budgets, and further outperforms oracle prefix-based baselines across models.

Limitations & Future Work A practical consideration of NUDGERL is the cost of generating strategy-level contexts. However, this is an offline process performed once prior to training, using a lightweight LLM (e.g., gpt-4o-mini), and the resulting contexts can be reused across training runs without additional overhead. A more fundamental limitation lies in how contexts are generated independently of the model being trained. The benefit of Context Forcing stems from inducing trajectories that are unlikely under the current policy. As training progresses, however, a fixed context pool may become less informative as the policy adapts. A promising direction for future work is model-adaptive context generation, which dynamically constructs contexts tailored to the current policy’s blind spots, potentially yielding more consistent exploration gains throughout training.

### References

- [1] Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https://matharena.ai/.
- [2] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [3] Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, and Ji-Rong Wen. Decomposing the entropy-performance exchange: The missing keys to unlocking effective reinforcement learning. arXiv preprint arXiv:2508.02260, 2025.
- [4] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.
- [5] Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, et al. Brorl: Scaling reinforcement learning via broadened exploration. arXiv preprint arXiv:2510.01180, 2025.
- [6] HuggingFace. Math-verify: A toolkit for verifying mathematical reasoning. https://github. com/huggingface/Math-Verify, 2024. Accessed 2026-05-06.
- [7] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [8] Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, and Jiang Bian. Self-hinting language models enhance reinforcement learning. arXiv preprint arXiv:2602.03143, 2026.
- [9] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.
- [10] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.
- [11] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv, 2503.20783, 2025. URL https://doi.org/10.48550/arXiv.2503.20783.
- [12] Mathematical Association of America. American mathematics competitions. https://www. maa.org/math-competitions, 2023.
- [13] Mathematical Association of America. Aime: American invitational mathematics examination. https://www.maa.org/math-competitions, 2025.
- [14] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott

Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL https:

//arxiv.org/abs/2512.13961.

- [15] OpenAI. Gpt-4o mini. https://openai.com/ko-KR/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024. Accessed: 2026-0504.
- [16] Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, and Aviral Kumar. Pope: Learning to reason on hard problems via privileged on-policy exploration. arXiv preprint arXiv:2601.18779, 2026.
- [17] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [18] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv, 2402.03300, 2024. URL https://doi.org/10.48550/arXiv. 2402.03300.
- [19] Yuda Song, Lili Chen, Fahim Tajwar, Remi Munos, Deepak Pathak, J Andrew Bagnell, Aarti Singh, and Andrea Zanette. Expanding the capabilities of reinforcement learning via text feedback. arXiv preprint arXiv:2602.02482, 2026.
- [20] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [21] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [22] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. TRL: Transformers Reinforcement Learning, 2020. URL https://github.com/huggingface/trl.
- [23] Fang Wu, Weihao Xuan, Ximing Lu, Zaïd Harchaoui, and Yejin Choi. The invisible leash: Why RLVR may not escape its origin. arXiv, 2507.14843, 2025. URL https://doi.org/10. 48550/arXiv.2507.14843.
- [24] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. arXiv, 2503.14476,

2025. URL https://doi.org/10.48550/arXiv.2503.14476.

- [25] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv, 2504.13837, 2025. URL https://doi.org/10.48550/arXiv.2504. 13837.
- [26] Xiaoyun Zhang, Xiaojian Yuan, Di Huang, Wang You, Chen Hu, Jingqing Ruan, Kejiang Chen, and Xing Hu. Rediscovering entropy regularization: Adaptive coefficient unlocks its potential for llm reinforcement learning. arXiv preprint arXiv:2510.10959, 2025.
- [27] Xuechen Zhang, Zijian Huang, Yingcong Li, Chenshun Ni, Jiasi Chen, and Samet Oymak. Bread: Branched rollouts from expert anchors bridge sft & rl for reasoning. arXiv preprint arXiv:2506.17211, 2025.

### A Related Work

#### A.1 Reinforcement Learning with Verifiable Rewards

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning abilities of large language models[9, 18, 20, 24]. By leveraging automatically verifiable signals, such as exact answers in mathematics or test-case correctness in code generation, RLVR enables effective policy optimization without dense human supervision.

A representative approach is Group-Relative Policy Optimization (GRPO) [18], which replaces value function estimation with group-wise comparisons among sampled rollouts, deriving advantages from relative reward differences within each group. Building on this formulation, subsequent work has introduced improvements such as decoupled clipping [24] and alternative normalization strategies [11] to enhance training stability.

These methods have been successfully applied across a range of reasoning tasks and model scales [9, 10], establishing RLVR as a standard post-training approach for LLMs. However, their effectiveness fundamentally depends on exploration: the policy can only improve on trajectories it has already sampled. As a result, insufficient exploration directly limits learning, making it a key bottleneck in RLVR. We next examine how prior work addresses this challenge.

#### A.2 Exploration in RLVR

- A straightforward approach to improving exploration is to scale the number of sampled rollouts. Prior work has shown that such rollout scaling can significantly improve performance by reducing the probability mass of un-sampled region[5]. However, this approach is computationally expensive and often impractical at scale.

More commonly, recent methods attempt to encourage exploration through objective design, such as entropy regularization [3, 26] or decoupled clipping [24]. While these approaches can steer the update toward exploration, they do not guarantee that useful or rare modes are actually sampled during training. In other words, shaping the distribution does not necessarily ensure coverage of meaningful trajectories, leaving exploration fundamentally limited.

Moreover, such distribution-level exploration is inherently stochastic and unconstrained, which can perturb the policy in semantically undesirable directions. Increasing entropy or aggressively reweighting probabilities may encourage the model to explore low-probability regions, but without any structural guidance, this often leads to incoherent or unproductive trajectories rather than meaningful reasoning strategies. As a result, these approaches lack control over how the policy explores, and fail to provide structured, strategy-level exploration that targets diverse and semantically valid modes of reasoning.

#### A.3 Usage of Privileged Information

Another key limitation of widely used group-based advantage methods, such as GRPO [18], is that they rely on relative comparisons within a group of rollouts. When all samples in a group are either correct or incorrect, these methods fail to provide informative learning signals.

To address this issue, recent works have introduced privileged information to assist the policy [8, 16, 19, 27], often in the form of oracle prefixes or intermediate solutions. These approaches improve the feasibility of solving hard problems by enabling the model to generate successful trajectories that would otherwise be unreachable.

However, such methods come with several limitations. First, privileged information is often difficult to scale, especially when it relies on oracle solutions or expensive annotations [16, 27]. Second, the mechanism by which the model internalizes this information and performs well without it at test time remains unclear [8]. Third, many approaches assume multi-turn or interactive settings [19], which may not align with standard single-turn RLVR setups.

More importantly, existing work primarily focuses on improving the feasibility of generating correct trajectories on difficult problems. In contrast, our work targets a complementary challenge: improving the diversity of exploration, even when successful trajectories are already attainable.

### B Details on Strategy Nudging

Strategy Generating Prompt. We use gpt-4o-mini to generate keyword-level hints for each problem. For the main experiments, we generate two hints per problem, while in the top-ranked setting, we first generate five candidate hints and select a subset based on oracle evaluation.

The exact prompt used for hint generation is as follows:

f"""Given the following math problem, generate {num_hints} different keyword hints that would help solve it. Each hint should be a specific mathematical concept, theorem, or technique (e.g., "Ceva’s theorem", "Lifting the exponents", "Triangle inequality").

Problem: {problem}

Please provide exactly {num_hints} hints in the following format (one hint per line, numbered):

- 1. [Hint 1]

- 2. [Hint 2]

... {num_hints}. [Hint {num_hints}]

Make sure each hint is a distinct mathematical concept or theorem."""

Strategy Nudging prompt. Given a problem and an optional hint, we construct prompts that encourage the model to follow a specific reasoning strategy. The model is instructed to explicitly separate its reasoning process and final answer using predefined delimiters.

reasoning_start = "<start_working_out>" reasoning_end = "<end_working_out>" solution_start = "<SOLUTION>" solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem. Think about the problem and provide your working out. Place it between {reasoning_start} and {reasoning_end}. Then, provide your solution between {solution_start}{solution_end}"""

def build_messages(problem: str, system_prompt: str, hint: str | None = None)

-> list[dict[str, str]]: context_block = "" if hint:

context_block = ( "Context (exploration condition):\n" f"- Use this hint/approach: {hint}\n\n" "Important:\n" "- Follow this approach as your primary strategy.\n\n"

)

user_content = ( "Problem:\n" f"{problem}\n\n" f"{context_block}" "Solve this step by step and provide your final numerical answer at the

end."

) return [

{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content},

]

Effect of Strategy Nudging. To evaluate the effect of Strategy Nudging, we sample 8 rollouts from Qwen3-4B-Instruct-2507 [21] on 200 problems from DAPO-17k-Processed [24], both with and without Strategy Nudging, and analyze the resulting rollout diversity via LLM-as-a-judge.

LLM-as-a-judge prompt. To analyze the diversity of generated rollouts, we employ an LLMas-a-judge using gpt-4o-mini to cluster solutions based on their underlying reasoning strategies and count the number of distinct solution modes. Given a problem and a set of rollouts, the model is instructed to identify the number of conceptually distinct solution approaches, while ignoring superficial differences such as phrasing or minor computational variations.

prompt = (f"Problem:\n{problem_text}\n\n" f"Here are {len(rollouts)} student solutions to this problem:\n" f"{formatted_rollouts}\n" f"Analyze these solutions and determine how many *conceptually distinct* solution strategies are used across them.\n"

f"Ignore minor calculation differences or phrasing variations. Focus on the core mathematical approach.\n" f"Provide the answer in the following format: ’Distinct Strategies: X’ where X is the integer count.\n"

f"Then briefly list the distinct strategies identified.") messages = [

{"role": "system", "content": "You are an expert math teacher evaluating

the diversity of student solution methods."}, {"role": "user", "content": prompt}

]

### C Details on Baselines

Rollout Scaling in GRPO. For controlled experiments, we scale the number of rollouts per prompt while adjusting the gradient accumulation steps and generation batch size accordingly, as summarized in Tab. 2. This ensures that the total optimization dynamics remain comparable across different rollout settings.

Implementing POPE [16]. To compare our method with oracle prefix-based approaches, we implement our own version of POPE [16]. We follow the original paper in using the same prompt format and dataset mixture (i.e., with and without privileged information). Since the length of oracle solutions varies across prior works [16, 27], we standardize this by truncating the oracle solution to 15% of its full length when used as a prefix.

Example of Generated Contexts. We provide an illustrative example of the strategy-level contexts used in our method. These contexts are lightweight, keyword-level hints that do not directly solve the problem, but instead steer the model toward distinct reasoning modes. Importantly, they are not intended to provide intermediate steps or solutions, but rather to act as high-level inductive biases that diversify exploration.

Problem

Let S be the set of triples (a, b, c) of non-negative integers such that a + b + c is even. The value of the sum

1 2a3b5c

(a,b,c)∈S

can be expressed as mn , where m and n are relatively prime positive integers. Compute m + n.

#### Oracle solution:

We compute the sum

1 2a3b5c

,

(a,b,c)∈S

where S consists of non-negative integer triples (a, b, c) with a + b + c even. Using the indicator

- 1

- 2

1 + (−1)a+b+c , we write

1a+b+c even =

 

  .

(−1)a+b+c 2a3b5c

1 2a3b5c

- 1

- 2

1 2a3b5c

=

+

a,b,c≥0

a,b,c≥0

(a,b,c)∈S

The first sum factors as

- 1

- 2a · b≥0

1 3b ·

1 5c

1 1 − 12

1 1 − 13

1 1 − 15

3 2 ·

5 4

15 4

·

·

= 2 ·

=

=

.

a≥0

c≥0

The second sum factors as

(−1)b 3b ·

(−1)c 5c

(−1)a 2a ·

1 1 + 13 ·

1 1 + 15

- 2

- 3 ·

1 1 + 12 ·

- 3

- 4 ·

- 5

- 6

5 12

=

=

=

.

a≥0

b≥0

c≥0

Thus, the total sum is

15 4

5 12

- 1

- 2 ·

50 12

25 12

- 1

- 2

+

=

=

.

Since 1225 is in lowest terms, we have m = 25, n = 12, and m + n = 37.

#### Strategy-level contexts(ours):

"Generating functions", "Parity of sums"

### D Training Detail

Framework. We used TRL [22] for implementing baselines and our algorithm.

Hyperparameters. The hyperparameters we used in training are given in Tab. 2.

Compute resources. For all experiments, we used NVIDIA H200 140GB GPUs.

### E Details on Evaluation

During evaluation, all hyperparameters are kept identical to Tab. 2, except for the temperature, which is set to 0.7.

### F Details on Case study

In this section, we provide qualitative examples from the case study presented in Fig. 4.

The GRPO-trained model predominantly relied on coordinate geometry combined with heuristic symmetry assumptions and case-by-case area decomposition. Although these approaches occasionally progressed toward partial solutions, they frequently resulted in excessively long derivations and truncated outputs before reaching the final answer.

In contrast, NUDGERL exploited the shoelace-formula strategy, which directly computes polygon areas from vertex coordinates. This strategy produced substantially shorter and more reliable reasoning trajectories, enabling successful completion within the generation budget.

Table 2: Hyperparameters for training.

Parameter Value LoRA rank 32 Max prompt length 2,048 Max completion length 6,144 RL steps 500 Batch size 4 Rollouts per prompt N Gradient accumulation steps N8 Generation batch size N Temperature 1 Min-p 0.0 Top-p 0.95 Top-k −1 Learning rate 2 × 10−5 LR scheduler cosine Weight decay 0.001 Warmup ratio 0.05 Optimizer AdamW (8-bit) KL coefficient β 0 Epsilon low 0.2 Epsilon high 0.2 (unless specified)

λ 1.1 λdistill 0.1 pdrop 0.5 Random seed 42

##### Problem

Let △ABC be a right triangle with ∠A = 90◦ and BC = 38. There exist points K and L inside the triangle such that

###### AK = AL = BK = CL = KL = 14.

The area of the quadrilateral BKLC can be expressed as n√3 for some positive integer n. Find n.

##### GRPO solution

<start_working_out> We are given a right triangle △ABC with ∠A = 90◦, and hypotenuse BC = 38. There are two points K and L inside the triangle such that:

###### AK = AL = BK = CL = KL = 14

We are to find the area of quadrilateral BKLC, which is expressed as n√3, and determine n.

—

- Step 1: Understand the configuration. Given that AK = AL = 14, and BK = CL = 14, and KL = 14, all these segments are equal to 14. So, we have:

- AK = AL = 14: So point K and L are both at distance 14 from A - BK = 14, CL = 14: So both K and L are at distance 14 from B and C, respectively - KL = 14 So, K lies at intersection of: - Circle centered at A with radius 14 - Circle centered at B with radius 14 Similarly, L lies at intersection of: - Circle centered at A with radius 14 - Circle centered at C with radius 14 And additionally, KL = 14 So both K and L lie on the intersection of circle centered at A and circles centered at B and C, respectively. Moreover, KL = 14, so the distance between K and L is 14.

—

- Step 2: Use symmetry. Since the triangle is right-angled at A, and the construction is symmetric (both K and L are symmetric with respect to A, and distances to B and C are equal), it is likely that the triangle is isosceles right triangle. Suppose AB = AC. Then △ABC is an isosceles right triangle with legs AB = AC = x, and hypotenuse BC = x√2 = 38 Then:

x

√

2 = 38 ⇒ x =

38 √2

= 19

√

2

So AB = AC = 19√2 Now, consider the point K such that AK = BK = 14 Similarly, L such that AL = CL = 14 Because of symmetry, if the triangle is symmetric about the angle bisector (the line y = x if we place A at origin), then K and L are symmetric points with respect to this line. So K lies in one region, L in the mirror region. Then KL = 14, and we can compute the distance between such symmetric points. Let’s try to place coordinates.

—

- Step 3: Coordinate setup. Place point A at origin (0, 0) Let B = (b, 0), C = (0, c) Since ∠A = 90◦, this is valid. We are told BC = 38 By distance formula:

BC2 = (b − 0)2 + (0 − c)2 = b2 + c2 = 382 = 1444 We also have points K and L such that:

- AK = 14 → K lies on circle x2 + y2 = 196 - BK = 14 → (x − b)2 + y2 = 196 - CL = 14 → x2 + (y − c)2 = 196 - AL = 14 → x2 + y2 = 196 - KL = 14 So both K and L lie on the circle centered at A with radius 14. Let’s find coordinates of K: intersection of circles: - x2 + y2 = 196 (1) - (x − b)2 + y2 = 196 (2) Subtract (1) from (2):

b 2

(x − b)2 − x2 = 0 ⇒ x2 − 2bx + b2 − x2 = −2bx + b2 = 0 ⇒ −2bx + b2 = 0 ⇒ x =

Plug into (1):

2

b2 4

b2 4

b 2

+ y2 = 196 ⇒

+ y2 = 196 ⇒ y2 = 196 −

Similarly, for point L, since CL = 14, and AL = 14, by symmetry, we get: x2 + (y − c)2 = 196, and x2 + y2 = 196

Subtract: (y − c)2 − y2 = 0 ⇒ −2cy + c2 = 0 ⇒ y = 2c Then from x2 + y2 = 196: x2 + 2 c 2 = 196 ⇒ x2 = 196 − c42 So coordinates:

- K = 2 b , yK , where yK2 = 196 − b42 - L = xL, 2c , where x2L = 196 − c42 Now, KL = 14 So compute distance between K and L:

2

b 2 − xL

c 2

2

KL2 =

+ yK −

###### = 196

This seems messy unless b = c, i.e., isosceles right triangle. Try assuming b = c Then b2 + c2 = 2b2 = 1444 ⇒ b2 = 722 ⇒ b = √722 = √722

Now, for K: xK = 2b , yK2 = 196 − b42 = 196 − 7224 = 196 − 180.5 = 15.5 Similarly, x2L = 196 − 7224 = 15.5, so xL = √15.5 So K = 2 b , √15.5 , L = √15.5, 2b Now compute KL2:

2

2

2

√

√

√

b 2 −

b 2

b 2 −

KL2 =

15.5 −

15.5

15.5

+

= 2

Set d = 2b , then b2 = 722 ⇒ d2 = 7224 = 180.5 Then KL2 = 2(d −

√15.5)2

Compute√15.5 ≈d3.=937√180.5 ≈ 13.43 Then d −

√15.5 ≈ 13.43 − 3.937 = 9.493 Then KL2 ≈ 2 × (9.493)2 ≈ 2 × 90.1 ≈ 180.2, which is not 196. But we need KL = 14, so KL2 = 196 So this value is too small — contradiction. Therefore, in the isosceles case, KL2 ≈ 180 < 196, so not satisfying the condition. So the triangle is not isosceles? But the symmetry suggests it might be. Alternatively, maybe the configuration implies symmetry, and the triangle is isosceles, but our calculation is wrong. Wait — perhaps we assumed the wrong point. Wait: we assumed that K is equidistant from A and B, so lies at x = b/2 But is that correct? Yes — from the two equations: x2 + y2 = 196 (x − b)2 + y2 = 196 Subtract: −2bx + b2 = 0 ⇒ x = b/2 Yes. Similarly, for L, y = c/2 So coordinates are correct.

Then KL2 = (b/2 − xL)2 + (yK − c/2)2 With x2L = 196 − c2/4, yK2 = 196 − b2/4 So define: Let u = b/2, v = c/2 Then b = 2u, c = 2v

- Then b2 + c2 = 4u2 + 4v2 = 1444 ⇒ u2 + v2 = 361 Now, yK2 = 196 − u2, so yK = √196 − u2

Similarly, x2L = 196 − v2, so xL = √196 − v2 Then:

KL2 = (u − 196 − v2)2 + ( 196 − u2 − v)2 = 196

This is symmetric in u and v, so likely u = v Try u = v Then u2 + v2 = 2u2 = 361 ⇒ u2 = 180.5 Then u = √180.5 ≈ 13.43 Then √196 − u2 = √196 − 180.5 = √15.5 ≈ 3.937 Then KL2 = 2(u −

√15.5)2 ≈ 2(13.43 − 3.937)2 = 2(9.493)2 ≈ 2 × 90.1 = 180.2 ̸= 196

Still too small. So even with symmetry, KL2 < 196 But we need KL2 = 196 So contradiction. Therefore, the triangle is not symmetric? But then why are AK = AL = BK = CL = 14? There is symmetry: K is symmetric to L in some way? Wait — maybe K and L are symmetric with respect to the angle bisector. But unless the triangle is isosceles, such symmetry is not possible. Alternatively, perhaps the points K and L are such that triangle AKL is equilateral? Given AK = AL = KL = 14, so triangle AKL is equilateral! Yes! This is key. We are told AK = AL = KL = 14, so △AKL is equilateral. So ∠KAL = 60◦ But ∠A = 90◦, so that’s possible. So point K and L are at distance 14 from A, and from each other, forming an equilateral triangle.

Also, BK = 14, CL = 14 So K lies on intersection of circles: center A, radius 14 and center B, radius 14 Similarly, L lies on intersection of circles: center A, radius 14 and center C, radius 14 So we can find coordinates. Let’s proceed with coordinate geometry. Let A = (0, 0), B = (c, 0), C = (0, b), so AB = c, AC = b, BC = √b2 + c2 = 38 So b2 + c2 = 1444 Now, point K lies at intersection of: - x2 + y2 = 196 (distance from A) - (x − c)2 + y2 = 196 (distance from B)

- Subtract: (x − c)2 − x2 = 0 ⇒ −2cx + c2 = 0 ⇒ x = 2c Then 2 c 2 + y2 = 196 ⇒ y2 = 196 − c42 Similarly, point L lies on: - x2 + y2 = 196 - x2 + (y − b)2 = 196

- Subtract: (y − b)2 − y2 = −2by + b2 = 0 ⇒ y = 2b

2

= 196 ⇒ x2 = 196 − b42

Then x2 + 2 b

So: - K = 2 c, yK , yK = 196 − c42 - L = xL, 2b , xL = 196 − b42 Now, KL = 14, so:

2

c 2 − xL

b 2

2

KL2 =

+ yK −

= 196

Let’s define: - u = 2c, v = 2b Now: - yK = √196 − u2 - xL = √196 − v2 Then:

- Then c = 2u, b = 2v, and b2 + c2 = 4u2 + 4v2 = 1444 ⇒ u2 + v2 = 361

KL2 = (u − 196 − v2)2 + ( 196 − u2 − v)2 = 196

Let’s expand this expression: First term: (u −

√196 − v2)2 = u2 − 2u√196 − v2 + (196 − v2) Second term: (√196 − u2 − v)2 = (196 − u2) − 2v√196 − u2 + v2 Add both: Total:

u2 + 196 − v2 + 196 − u2 + v2 − 2u 196 − v2 − 2v 196 − u2

= 392 − 2u 196 − v2 − 2v 196 − u2. Set equal to 196:

392 − 2u 196 − v2 − 2v 196 − u2 = 196 2u 196 − v2 + 2v 196 − u2 = 296

u 196 − v2 + v 196 − u2 = 148.

Now, recall that u2 + v2 = 361 Let S = u√196 − v2 + v√196 − u2 = 148 We can try to assume symmetry: u = v Then 2u2 = 361 ⇒ u2 = 180.5 Then u = √180.5 ≈ 13.43 Then √196 − u2 = √196 − 180.5 = √15.5 ≈ 3.937 Then S = u ·

√15.5 + v ·

√15.5 = 2u ·

√15.5 ≈ 2 × 13.43 × 3.937 ≈ 2 × 52.9 ≈ 105.8 ̸= 148

Too small. So not symmetric. We need to solve: u2 + v2 = 361 u√196 − v2 + v√196 − u2 = 148 Try to make a substitution. Let a = u2, b = v2, so a + b = 361 Then u = √a, v =

√

b

So√athe√196equation− b +becomes:

√

b√196 − a = 148 Note: 196 − b = 196 − (361 − a) = 196 − 361 + a = a − 165 Similarly, 196 − a = b − 165 So:

a(a − 165) + b(b − 165) = 148 But b = 361 − a, so:

a(a − 165) + (361 − a)(361 − a − 165) = 148 Simplify:

a(a − 165) + (361 − a)(196 − a) = 148

Now, define function: f(a) = a(a − 165) + (361 − a)(196 − a) We want f(a) = 148 Domain: a > 165, and 361 − a > 0 ⇒ a < 361, and 196 − a > 0 ⇒ a < 196 So 165 < a < 196 Try a = 180 Then:

- -√a2896(a −≈165)53.82= -180Sum×≈1551.=962700+ 53,.82√2700= 105≈.7851→.96too- (361small− 180)(196 − 180) = (181)(16) = 2896, Try a = 190

- -√a1026(a −≈165)32.03=-190Sum×≈2568.=92 +475032.,03√4750= 100≈.9568→.92even- (361smaller− 190)(196 − 190) = (171)(6) = 1026, Wait — it’s decreasing? Wait, when a increases, first term increases, second decreases. But at a = 165, first term is 0, second term: (361 − 165)(196 − 165) = (196)(31) = 6076, √6076 ≈ 77.95 At a = 196, first term: 196(31) = 6104, √6104 ≈ 78.13 Second term: (361 − 196)(0) = 0 So maximum value of f(a) is about 78.13 + 0 = 78.13? Wait — this is less than 148! But we need sum = 148 But the maximum value of each square root is about 78, so sum can’t exceed about 156? Wait: a(a − 165), for a = 196: 196 × (196 − 165) = 196 × 31 = 6104, √6104 ≈ 78.13 Similarly, maximum of other term at a = 165 is √196 × 31 = √6076 ≈ 77.95 So sum max is about 78 + 78 = 156 So 148 is possible! But at a = 190, we got only ∼ 101 — why? Wait: a = 190, a(a − 165) = 190 × 25 = 4750, √4750 ≈ 68.9 (361 − 190) = 171, (196 − 190) = 6, 171 × 6 = 1026, √1026 ≈ 32.0 Sum ≈ 100.9 At a = 180: 51.96 + 53.82 = 105.78 At a = 170: a(a − 165) = 170 × 5 = 850, √850 ≈ 29.15 (361 − 170) = 191, (196 − 170) = 26, 191 × 26 = 4966, √4966 ≈ 70.47 Sum ≈ 29.15 + 70.47 = 99.62 All are below 100! Wait — this can’t be. The maximum of the sum is about 156, but values are all below 100. I must have made a calculation error. Wait: when a = 196, first term: a(a − 165) = 196 × (196 − 165) = 196 × 31 = 6104, √6104 =?

√6104 ≈ 78.13 Second term: (361 − 196)(196 − 196) = 165 × 0 = 0 → sum = 78.13 At a = 165: first term = 0, second term: (196)(31) = 6076, sqrt ≈ 77.95 → sum = 77.95 So the maximum is about 78.13 But we need sum = 148 Impossible! Contradiction. So there is no solution? But the problem states that such points exist. Therefore, our assumption must be wrong. Wait — is it possible that the point K is not the intersection of circle A and circle B, but another point? The given is: AK = AL = BK = CL = KL = 14 We have:

6104: 782 = 6084, 792 = 6241 →

- AK = 14 - BK = 14 - AL = 14 - CL = 14 - KL = 14 So yes, K is at distance 14 from both A and B, so lies on intersection

##### NudgeRL Solution

<start_working_out> We are given a right triangle △ABC with ∠A = 90◦ and hypotenuse BC = 38. There are two points K and L inside the triangle such that:

###### AK = AL = BK = CL = KL = 14

We are to find the area of quadrilateral BKLC, which is expressed as n√3, and find n.

—

- ### Step 1: Understand the configuration We have: - ∠A = 90◦, so AB and AC are legs, BC = 38 is the hypotenuse. - Points K and L are inside the triangle. - All of the following distances are equal to 14: - AK = AL = BK = CL = KL = 14 This suggests symmetry. Since AK = AL, and BK = CL, and KL = 14, it’s likely that K and L are symmetric with respect to the angle bisector or the median from A. Moreover, AK = AL suggests that K and L lie on the perpendicular bisector of BL or something symmetric. But let’s try to exploit symmetry.

—

- ### Step 2: Consider symmetry Since AK = AL and BK = CL, and the triangle is right-angled at A, a natural symmetry is about the angle bisector of ∠A, which in a right triangle is the line y = x if we place A at the origin and AB, AC along the axes. Let’s place the triangle in coordinate plane: Let: - A = (0, 0) - B = (b, 0) - C = (0, c) Then BC = 38, so by Pythagoras:

b2 + c2 = 382 = 1444

We are to find points K and L such that: - AK = AL = 14 - BK = 14 - CL = 14 - KL = 14 Let’s suppose that due to symmetry, K and L are symmetric with respect to the line y = x. Then: Let K = (x, y), then L = (y, x)

Then: - AK = x2 + y2 = 14 ⇒ x2 +y2 = 196 — (1) - BK = (x − b)2 + y2 = 14 ⇒ (x−b)2 +y2 = 196 — (2) - CL = (y − c)2 + x2 = 14 ⇒ (y − c)2 + x2 = 196 — (3) - KL = (x − y)2 + (y − x)2 =

2(x − y)2 = √2|x − y| = 14 So:

√

2|x − y| = 14 ⇒ |x − y| =

14 √2

= 7

√

2

- So x − y = ±7√2. Without loss of generality, assume x > y, so x − y = 7√2 — (4)

—

- ### Step 3: Use equations (1), (2), (3), (4) From (1): x2 + y2 = 196 From (4): x = y + 7√2 Plug into (1):

√

2)2 + y2 = 196 y2 + 14

(y + 7

√

2y + 98 + y2 = 196 2y2 + 14

√

2y + 98 = 196 2y2 + 14

√

2y = 98 y2 + 7

√

2y = 49

This is a quadratic in y. Let’s solve it. Complete the square:

√

y2 + 7

2y = 49

7√2 2

7√2 2

2

2

√

y2 + 7

2y +

= 49 +

√

49 · 2 4

98 4

y2 + 7

= 49 + 24.5 = 73.5 So:

= 49 +

2y +

7√2 2

2

147 2 So:

y +

= 73.5 =

7√2 2 ±

147 2 This is messy, and we haven’t yet used the other conditions involving b and c. So perhaps we should go back to equations (2) and (3).

y = −

From equation (2): (x − b)2 + y2 = 196 From (1): x2 + y2 = 196 Subtract (1) - (2):

(x2 + y2) − [(x − b)2 + y2] = 0 x2 − (x2 − 2bx + b2) = 0 x2 − x2 + 2bx − b2 = 0

2bx = b2

- So b = 2x Similarly, go to equation (3): (y − c)2 + x2 = 196 From (1): x2 + y2 = 196 Subtract (3) - (1):

[(y − c)2 + x2] − [x2 + y2] = 0

(y − c)2 − y2 = 0 y2 − 2cy + c2 − y2 = 0

−2cy + c2 = 0

- So c = 2y Now we have: - b = 2x - c = 2y - x2 + y2 = 196 - x − y = 7√2 And also from the hypotenuse:

b2 + c2 = 1444 (2x)2 + (2y)2 = 1444

4x2 + 4y2 = 1444 x2 + y2 = 361

Wait! This contradicts equation (1), which said x2 + y2 = 196. So contradiction! That means our assumption that K = (x, y), L = (y, x) is invalid? But why? Because AK = AL = 14, and symmetry about y = x makes sense, but the distance from K to B and L to C may not match unless the legs are in a certain ratio. But we get a contradiction: x2 + y2 = 196 from AK = 14, but from b2 + c2 = 1444, we get x2 + y2 = 361, which is not equal. So our assumption must be wrong. Where did we go wrong? Ah — in the step where we said from (2) and (1), we got b = 2x. Let’s double-check:

- From (1): x2 + y2 = 196
- From (2): (x − b)2 + y2 = 196 Subtract (1) from (2):

(x − b)2 − x2 = 0 x2 − 2bx + b2 − x2 = 0 −2bx + b2 = 0 b(b − 2x) = 0

So either b = 0 (impossible) or b = 2x Same for c = 2y So the algebra seems correct. But then b2 + c2 = 4x2 + 4y2 = 4(x2 + y2) = 4 × 196 = 784 But we are told BC = 38, so b2 + c2 = 382 = 1444 But 784 ̸= 1444 Contradiction. So our assumption that K and L are symmetric across y = x must be invalid. Alternative idea: Maybe K and L are symmetric with respect to the altitude from A, or the angle bisector? Alternatively, perhaps the configuration is such that K and L are located so that they are equidistant from A, B, C, and each other. Wait — we are given:

- AK = AL = 14 - BK = 14 - CL = 14 - KL = 14 So: - K is at distance 14 from A and B - L is at distance 14 from A and C - K and L are at distance 14 from each other

So K lies at intersection of circles: - Circle centered at A with radius 14 - Circle centered at B with radius 14 Similarly, L lies at intersection of: - Circle centered at A with radius 14 - Circle centered at C with radius 14 So both K and L are at intersection of circle A and circle B, and A and C respectively. So we can find coordinates.

—

- ### Step 4: Coordinate geometry setup Let’s place: - A = (0, 0) - B = (b, 0) - C = (0, c) Then b2 + c2 = 382 = 1444 Now, find intersection of circle centered at A(0, 0) with radius 14 and circle centered at B(b, 0) with radius 14.

- Equation of circle at A: x2 + y2 = 196
- Equation of circle at B: (x − b)2 + y2 = 196 Subtract:

(x − b)2 − x2 = 0 x2 − 2bx + b2 − x2 = 0

−2bx + b2 = 0 x =

b 2

Plug into x2 + y2 = 196:

2

b 2

+ y2 = 196 b2

+ y2 = 196

4

b2 4

y2 = 196 −

- So y = ± 196 − b42 Since K is inside the triangle, we take y > 0, so

b2 4

yK = 196 −

Similarly, find L: intersection of circle at A and circle at C(0, c) Circle at C: x2 + (y − c)2 = 196 Subtract from x2 + y2 = 196:

x2 + y2 − [x2 + (y − c)2] = 0 y2 − (y2 − 2cy + c2) = 0 2cy − c2 = 0

c 2

y =

Then:

c2

c 2

2

x2 +

= 196 ⇒ x2 = 196 −

4 So:

c2 4

xL = 196 −

(positive, since inside triangle) So: - K = 2 b , 196 − b42 - L = 196 − c42 , 2c We are given that KL = 14 So compute distance between K and L:

  b

 

  196 −

 

2

2

c2 4

b2 4 −

c 2

KL2 =

###### = 142 = 196

2 − 196 −

+

###### Let’s define: - p = 2b - q = 2c Then b = 2p, c = 2q

Then:

###### b2 + c2 = 4p2 + 4q2 = 4(p2 + q2) = 1444 ⇒ p2 + q2 = 361

Now express the coordinates: - K = (p, 196 − p2) - L = ( 196 − q2, q) Then:

###### KL2 = (p − 196 − q2)2 + ( 196 − p2 − q)2 = 196

Expand: First term:

(p − 196 − q2)2 = p2 − 2p 196 − q2 + (196 − q2) Second term:

( 196 − p2 − q)2 = (196 − p2) − 2q 196 − p2 + q2 Add both:

p2 + 196 − q2 + 196 − p2 + q2

− 2p 196 − q2 − 2q 196 − p2 = 196 Simplify:

- p2 − p2 = 0 - −q2 + q2 = 0 - 196 + 196 = 392 So:

392 − 2p 196 − q2 − 2q 196 − p2 = 196 Bring to left:

392 − 196 = 2p 196 − q2 + 2q 196 − p2 196 = 2p 196 − q2 + 2q 196 − p2

98 = p 196 − q2 + q 196 − p2

Now recall that p2 + q2 = 361 Let’s define x = p, y = q, so: - x2 + y2 = 361 - x 196 − y2 + y√196 − x2 = 98 Let’s try to simplify or guess values. Note that x2 + y2 = 361, so x2, y2 ≤ 361 Also:

196 − x2 ≥ 0 ⇒ x2 ≤ 196

similarly y2 ≤ 196 Let’s suppose x = y. Then:

###### 2x2 = 361 ⇒ x2 = 180.5 ⇒ x ≈ 13.43

Check if the equation holds: Left side:

x 196 − x2 + x 196 − x2 = 2x 196 − x2 Since:

x2 = 180.5 ⇒ 196 − x2 = 15.5 We have: √

15.5 ≈ 3.937 and

√

180.5 ≈ 13.43 So:

x ≈

2 × 13.43 × 3.937 ≈ 2 × 52.9 ≈ 105.8 ̸= 98

Too big. So x = y is too big. We need smaller value. Try to assume x2 = a, y2 = 361 − a

We want: √a 196 − (361 − a) + √361 − a√196 − a = 98 Simplify:

196 − (361 − a) = a − 165 So:

a(a − 165) + (361 − a)(196 − a) = 98

Now define function:

f(a) = a(a − 165) + (361 − a)(196 − a) Domain:

a > 165 since:

a(a − 165) ≥ 0 Also:

a ≤ 196, 361 − a ≥ 0 ⇒ a ≤ 361 so:

a ∈ (165, 196] Try a = 169:

-√a5184(a −=165)72 = 169(4) = 676, √676 = 26 - 361 − 169 = 192 - 196 − 169 = 27 - 192 × 27 = 5184 Sum:

26 + 72 = 98

Perfect! So:

a = 169 Hence:

x2 = 169 ⇒ x = 13 and:

√

√

y2 = 361 − 169 = 192 ⇒ y =

3 Now check:

192 = 8

x2 + y2 = 169 + 192 = 361 Also:

√

√

a(a − 165) =

169 · 4 =

676 = 26 and:

√

√

(361 − 169)(196 − 169) =

192 · 27 =

5184 = 72 Sum:

26 + 72 = 98

Perfect. So: - p = x = 13 - q = y = √192 = 8√3 Then: - b = 2p = 26 - c = 2q = 16√3 Check:

√

b2 + c2 = 262 + (16

3)2 = 676 + 256 · 3 = 676 + 768 = 1444 = 382

Good. Now we have coordinates:

  b

  = (13, √196 − 169) = (13,

2

√

√

b 2

, 196 −

K =

27) = (13, 3

3)

2

and:

c 2

c 2

2

L = 196 −

,

Since: c 2

√

= 8

3 and:

√

c 2

2

3)2 = 192

= (8

So: √196 − 192 =

√

4 = 2 Thus:

√

L = (2, 8

3)

Therefore: - K = (13, 3√3) - L = (2, 8√3) Now we want the area of quadrilateral BKLC Points: - B = (26, 0) - K = (13, 3√3) - L = (2, 8√3) - C = (0, 16√3) Quadrilateral BKLC — in order:

B → K → L → C → B

We can compute its area using shoelace formula. List the points in order:

1. B = (26, 0) 2. K = (13, 3√3) 3. L = (2, 8√3) 4. C = (0, 16√3) Back to B = (26, 0) Shoelace formula:

- 1

- 2

Area =

xiyi+1 − yixi+1 Compute:

xiyi+1

- 26 · 3√3 = 78√3 - 13 · 8√3 = 104√3 - 2 · 16√3 = 32√3 - 0 · 0 = 0 Sum:

√

√

3 Compute:

(78 + 104 + 32)

3 = 214

yixi+1

- 0 · 13 = 0 - 3√3 · 2 = 6√3 - 8√3 · 0 = 0 - 16√3 · 26 = 416√3 Sum:

√

√

(0 + 6 + 0 + 416)

3 Now:

3 = 422

√

√

√

√

- 1

- 2|214

- 1

- 2 · 208

Area =

3 − 422

3| =

3 So:

3 = 104

n = 104

Final Answer: <SOLUTION>104</SOLUTION>

### G Full Evaluation Results

- Table 3: pdrop ablation results. We report pass@1 estimated using 128 rollouts. Best results are represented as bold.

pdrop AIME24 AIME25 AMC23 MATH500 Average

0.00 0.418 0.344 0.759 0.628 0.537 0.25 0.461 0.354 0.773 0.658 0.561 0.50 0.482 0.393 0.857 0.660 0.598 0.75 0.458 0.361 0.796 0.649 0.566

- Table 4: Hint sampling ablation results. We report pass@1 estimated using 128 rollouts. Best results are represented as bold.

Sampling AIME24 AIME25 AMC23 MATH500 Average

Random 0.482 0.393 0.857 0.660 0.598 Top ranked 0.448 0.355 0.774 0.632 0.552

### H Broader Impacts

This paper proposes an efficient framework for structured exploration in reinforcement learning with verifiable rewards (RLVR). On the positive side, our method improves exploration efficiency without

- Table 5: λ ablation results. We report pass@1 estimated using 128 rollouts. Best results are represented as bold.

λ AIME24 AIME25 AMC23 MATH500 Average

- 0.9 0.403 0.366 0.806 0.648 0.556
- 1.0 0.436 0.359 0.831 0.643 0.567 1.1 0.482 0.393 0.857 0.660 0.598

- Table 6: λdistill ablation results. We report pass@1 estimated using 128 rollouts. Best results are represented as bold.

λdistill AIME24 AIME25 AMC23 MATH500 Average

- 0.0 0.423 0.362 0.826 0.628 0.560
- 0.1 0.482 0.393 0.857 0.660 0.598 0.5 0.425 0.361 0.730 0.629 0.536

relying on extremely large rollout budgets or expensive oracle supervision, which may help reduce the computational cost of training reasoning models and improve accessibility for smaller research groups.

However, improving exploration efficiency may also contribute to the development of increasingly capable reasoning systems, which could be misused in harmful or unintended ways. We therefore emphasize the importance of continued research on safety, oversight, and responsible deployment.

- Table 7: ϵhigh scaling results. We report pass@1 estimated using 128 rollouts. Best results are represented as bold.

Algorithm ϵhigh AIME24 AIME25 AMC23 MATH500 Average NudgeRL 0.2 0.482 0.393 0.857 0.660 0.598

0.2 0.444 0.367 0.749 0.668 0.557 0.24 0.451 0.373 0.795 0.645 0.566 0.28 0.443 0.372 0.793 0.648 0.564 0.32 0.452 0.358 0.813 0.640 0.566 0.36 0.432 0.338 0.845 0.647 0.565 0.40 0.406 0.341 0.781 0.638 0.541

GRPO

