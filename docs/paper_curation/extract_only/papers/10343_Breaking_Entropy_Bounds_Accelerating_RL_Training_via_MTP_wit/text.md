[Figure 1]

2026-06-11

## Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling

Yucheng Li† Huiqiang Jiang†‡ Yang Xu Jianxin Yang Yi Zhang Yizhong Cao Yuhao Shen Fan Zhou Rui Men Jianwei Zhang An Yang Bowen Yu Bo Zheng Fei Huang Junyang Lin Dayiheng Liu Jingren Zhou

Qwen Team, Alibaba Inc.

# arXiv:2606.12370v1[cs.LG]10Jun2026

#### Abstract

Reinforcement learning (RL) has become a key component in modern large language models, yet the rollout stage remains the key bottleneck in RL training pipelines. Although Multi-Token Prediction (MTP) offers a natural solution to accelerate rollouts through speculative decoding, many studies have observed that MTP acceptance rates degrade significantly during RL training, leading to limited speedup performance. To address this bottleneck, we present Bebop, a systematic study of MTP in LLM post-training, and offer practical recipes to integrate MTP into large-scale RL pipelines. First, we reveal that the MTP acceptance rate is fundamentally bounded by the fluctuation of model entropy, which demonstrates a clear negative linear relationship with the rise of entropy in the RL stage (§3). Second, we show that probabilistic rejection sampling largely alleviates the disturbance introduced by entropy in RL compared to greedy draft sampling. We further identify that the conventional MTP training objectives (cross-entropy or KL) are suboptimal in such settings, and therefore we propose a novel end-to-end TV loss that directly optimizes multi-step rejection sampling acceptance rate, yielding ∼10% acceptance rate improvements, achieving up to 95% acceptance rates and up to 25% extra inference throughput gains across mathematical reasoning, code generation, and agentic tasks (§4). Third, we test various online MTP training strategies during RL and show that pre-RL MTP training with e2e TV loss and rejection sampling achieves a consistent acceptance rate and speedup throughout the entire RL, eliminating the need for costly online MTP updating (§5). We provide extensive experiments and analysis that validate our findings. Experimental results show our method achieves up to 1.8× end-to-end acceleration in async RL training of Qwen3.5, Qwen3.6, and Qwen3.7 models.

Target Only

Target p TV Draft CE Draft

3.8

1.2

Narrow

RS w/ CE Loss RS w/ TV Loss

TV Overlap CE Overlap

3.6

1.0

TV draft

3.4

0.8

ProbabilityDensity

AcceptLength

TV Accept 85%

3.2

0.6

CE draft

3.0

0.4

2.8

CE Accept 74%

0.2

2.6

0.0

0.1 0.2 0.3 0.4 0.5 0.6

-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0

Entropy Loss

Token / Logit Space

(a) Entropy vs. Accept Length

(b) Draft/Target Distribution

Figure 1: (a) MTP acceptance rates degrade linearly with policy entropy fluctuation in RL; training MTP with our novel e2e TV loss largely eliminates this entropy dependence under rejection sampling. Each point represents the mean entropy and accept length at one RL step across different-size Qwen3.5, 3.6 and 3.7 training runs in various tasks. (b) The TV-trained MTP achieves substantially better distributional overlap with the policy model, yielding superior acceptance rate and speedup.

†Equal contribution. ‡ Corresponding author.

#### 1 Introduction

Reinforcement learning (RL) has become a key paradigm in modern large language model (LLM) training (OpenAI, 2026; Anthropic, 2026; Qwen Team, 2026b; DeepSeek-AI, 2026; GLM Team, 2026; Kimi Team, 2026; MiniMax, 2026a). However, RL training for LLMs is computationally expensive, with the end-to-end time heavily dominated by inference rollouts in both single- and multi-turn settings. Although recent progress in asynchronous RL frameworks (Fu et al., 2025; Wang et al., 2025; THUDM, 2025) can partially alleviate long-tail latency issues, rollout costs remain the primary bottleneck in RL training. Multi-Token Prediction (MTP) has recently gained prominence as a scalable speculative decoding paradigm to accelerate LLM inference (DeepSeek-AI, 2024; Qwen Team, 2026a). This naturally raises the question: can MTP be effectively leveraged to accelerate RL training for LLMs?

We conduct extensive experiments and show that using MTP directly in RL training often suffers from a significant decline in acceptance rates and therefore leads to limited speedup. Specifically, there are two factors that may affect MTP acceptance rates during RL: 1) to encourage exploration, the policy model often maintains a rather large entropy–or even shows a gradually increasing entropy curve, which makes it harder to predict draft tokens, degrading the acceptance rate; 2) the weight updates of the policy model cause distribution mismatch between the policy model and the MTP module (frozen in RL training), that may affect the acceptance rate. Through our theoretical analysis and empirical decomposition (§3), we show that entropy is the dominant factor driving acceptance rate degradation, while the mismatch introduced by policy updates remains negligible (Fig. 3). To tackle the entropy bound challenge and ensure the speedup of MTP, recent works (Chen et al., 2026b; Li et al., 2025; MiniMax, 2026b) have proposed online MTP training during RL to mitigate this degradation, yet this approach introduces significant memory and latency overhead and yields limited improvements in many RL tasks.

In this paper, we introduce Bebop1 and show that using probabilistic rejection sampling2 instead of the common greedy target-only sampling3 largely mitigates the acceptance rate degradation driven by policy entropy fluctuation (§3.3) and provides a large improvement in acceptance rate. The key insight is that target-only acceptance is fundamentally capped by maxy p(y), which decreases directly as entropy rises, whereas rejection sampling acceptance equals the full distributional overlap ∑v min(p(v), q(v)) and is therefore much less sensitive to entropy shifts. We further identify that existing MTP training objectives, such as cross-entropy (CE) or KL divergence, are suboptimal for rejection sampling: CE/KL only indirectly improve the distributional overlap that determines rejection sampling acceptance. This motivates us to propose a novel end-to-end TV loss that optimizes the joint multi-step overlap that directly improves rejection sampling acceptance rate.

Bebop produces MTP models that maintain consistent acceptance rates throughout the entire RL training process. These rates remain largely invariant to entropy changes. Bebop achieves this stability using only a lightweight pre-RL MTP training phase with an e2e TV loss, paired with rejection sampling during rollouts, eliminating the need for MTP co-training during RL.

Specifically, we make the following contributions:

- • Entropy Constraints on MTP Acceptance (§3). We show that MTP acceptance rates are fundamentally constrained by the target model’s entropy in RL training, exhibiting a clear negative linear relationship across diverse tasks and models. We further show that rejection sampling largely improves the acceptance rate in RL, as its acceptance depends on policy-draft overlap and is less sensitive to entropy shifts.
- • End-to-End TV Loss for MTP Training (§4). We identify that CE/KL-trained MTP produces suboptimal results in rejection sampling, and thereby introduce a novel end-to-end TV loss that directly optimizes the multi-step rejection sampling acceptance rate. We show that the e2e TV loss ensures stable training, produces inherently entropy-invariant MTP, and yields an extra ∼10% improvement in acceptance rate.
- • MTP Adaptation Strategy for RL (§5). We show that with a lightweight pre-RL MTP training with e2e TV loss and rejection sampling, our MTP module provides consistent acceptance rates throughout the entire RL training. The other factor, policy-draft mismatch driven by policy updates, is negligible, which eliminates the need for costly MTP online training during RL.
- • Extensive Empirical Validation and Analysis (§6, §7). Through large-scale experiments with

1Breaking Entropy Bounds for Optimal Prediction 2We have released our implementation at https://github.com/sgl-project/sglang/pull/26312. 3Target-only sampling means the verification of speculative decoding uses only target probability without caching

draft probability: it selects draft tokens via argmax and uses 1 as q(yˆ) in the verification.

Qwen3.5, 3.6, and 3.7 models on reasoning, coding, and various agentic tasks, we validate Bebop and provide practical recipes for integrating MTP into RL pipelines, achieving up to 1.8× end-toend acceleration of async RL pipelines. We further analyze how TV loss shapes draft distributions, the robustness of rejection sampling under policy updates, and the effects of temperature and generation length on acceptance rates.

#### 2 Preliminaries

###### 2.1 Multi-Token Prediction and Speculative Decoding

As an effective paradigm of speculative decoding (Leviathan et al., 2023; Chen et al., 2023), Multi-Token Prediction (MTP) augments autoregressive LLMs with lightweight draft heads that sequentially predict multiple future tokens (Gloeckle et al., 2024; DeepSeek-AI, 2024; Yang et al., 2025). Let p(·|x, y<t) denote the target (backbone) model’s next-token distribution at position t, and q(·|x, y<t) denote the draft head’s predicted distribution. During inference, MTP operates in a draft-then-verify paradigm: a chain of γ draft heads sequentially proposes candidate tokens yˆt+1, . . . , yˆt+γ, where each head takes the previous head’s hidden state as input; the γ candidates are then verified against the target model in a single forward pass.

The expected number of accepted tokens per verification step, which we call the acceptance length, directly determines the inference throughput. This acceptance length depends on the specific acceptance methods used during verification, detailed in the following section.

###### 2.2 Acceptance Methods

In speculative decoding, two acceptance methods are commonly used: Target-Only Sampling and Rejection Sampling. Fig. 13 illustrates the acceptance rate distributions of representative models under each method.

Target-Only Sampling. Under target-only sampling, the draft token is selected greedily as yˆ = argmaxy q(y) and accepted with probability p(yˆ), using only the target model’s probability. The singlestep acceptance rate is:

αTO = p(yˆ) = p argmax

q(y) . (1)

y

If rejected, the output token is resampled from the residual distribution presid(y) ∝ p(y) [y ̸= yˆ], ensuring the overall output distribution remains unbiased. Notably, for draft models with relatively low acceptance rates, target-only sampling can yield higher throughput than rejection sampling, as the simpler acceptance criterion avoids the overhead of caching and computing the draft probability vectors.

Rejection Sampling. Under rejection sampling (Leviathan et al., 2023; Chen et al., 2023), a draft token yˆ ∼ q(·) is accepted with probability min(1, p(yˆ)/q(yˆ)). The expected single-step acceptance rate is:

- p(yˆ)

- q(yˆ)

αRS = Eyˆ∼q min 1,

= ∑

min p(y), q(y) = 1 − dTV(p, q), (2)

y

where dTV(p, q) = 12 ∑y |p(y) − q(y)| is the Total Variation distance (Levin and Peres, 2017). This method provides an unbiased guarantee: the output distribution is exactly the target distribution p, regardless of the draft quality.

###### 2.3 Reinforcement Learning for LLMs

We consider the standard RL framework for LLMs, where a policy πθ (the LLM) generates trajectories y to prompts x ∼ D and receives scalar rewards R(x, y). We adopt GRPO (Shao et al., 2024), which samples

a group of G trajectories {y1, . . . , yG} from the rollout policy πθold for each prompt, and optimizes the clipped surrogate objective:

|yi|

G

1 G

1 |yi|

min ri,t Aˆi, clip(ri,t,1−ϵ,1+ϵ) Aˆi , (3)

### ∑

### ∑

J (θ) = Ex∼D

t=1

i=1

where ri,t = πθ(yi,t|x, yi,<t)/πθold(yi,t|x, yi,<t) is the importance sampling ratio and Aˆi = (R(x, yi) − µG)/σG is the group-normalized advantage.

- 0.958

- 0.960

AcceptanceRate

MTP Step 1

0 50 100 150 200 250 300 350

Training Step

0.870

0.875

0.880

0.885

0.890

0.895

MTP Step 2

0 50 100 150 200 250 300 350

Training Step

0.78

0.79

0.80

0.81

MTP Step 3

- Figure 2: Per-step MTP acceptance rates during SWE-bench RL training with Qwen3.5-3.6 Plus. Each line represents a separate RL run. Later MTP steps exhibit progressively larger degradation: step 1 drops by

- 1.2%, step 2 by 2.6%, and step 3 by 3.5% over the course of training.

RL training for LLMs typically operates in a loop of three stages: (1) rollout uses the current policy to generate trajectories in an inference engine, potentially involving multi-turn sandbox or tool interactions; (2) reward evaluates these generated trajectories with a reward model or verifier; and (3) update optimizes the policy inside a training engine using policy gradient methods. The asynchronous RL or partial rollout frameworks are commonly adopted to mitigate the bubble overhead caused by long-tail trajectories during rollout (Fu et al., 2025; Wang et al., 2025; THUDM, 2025; Qin et al., 2025; MiniMax, 2026b). Despite asynchronous designs, the rollout stage remains the dominant computational bottleneck. While MTP offers a powerful acceleration paradigm to alleviate this burden, its direct application in RL environments exposes unique performance gaps that require further optimization.

- 2.4 Degradation of MTP During RL Training

During RL training, MTP acceptance rates degrade significantly across prediction steps. As shown in Fig. 2, later steps experience progressively larger drops. The per-step acceptance rate decline ranges from 1.2% at step 1 to 3.5% at step 3.

Recent work (MiniMax, 2026b; Chen et al., 2026b; Li et al., 2025) primarily attributes this degradation to distribution mismatch. Specifically, a gap emerges between the static draft predictions q = qϕ(· | x, y<t) and the evolving target distribution p = πθ(· | x, y<t) because backbone weight updates leave the draft heads behind. While this mismatch exists, we argue that this perspective is incomplete. We identify shifts in the target model’s entropy H(p) during RL training as another fundamental driver. These entropy shifts inherently alter the achievable acceptance bounds regardless of draft accuracy. These two factors compound through the multi-step acceptance structure:

- 1. Single-step degradation: The per-token acceptance rate αi continuously decreases as the TV distance dTV(p, q) grows, driven by the persistent divergence between the draft-target distribution.
- 2. Multi-step compounding: For γ-step MTP, the expected acceptance length involves products of per-step acceptance rates, so degradation compounds multiplicatively: E[L] = ∑γj=1 ∏ij=1 αi.

Crucially, our decomposition analysis in §3 and Fig. 3 challenges the conventional mismatch-centric view. We demonstrate that the entropy-driven component actually dominates the acceptance rate fluctuation during RL training. The distribution mismatch component remains comparatively small. This key insight reshapes our understanding of MTP degradation and directly motivates our subsequent optimization strategy.

3 Target Entropy Constraints on MTP Acceptance

In this section, we analyze how the target model’s entropy fundamentally constrains MTP acceptance rates, which explains the acceptance rate degradation driven by entropy shifts during RL training. This further motivates our training objectives in §4.

- 3.1 Formulation

0.956

0.954

0.952

0.950

0 50 100 150 200 250 300 350

Training Step

Consider a fixed position t in the generation process. Let p ∈ ∆|V| denote the target model’s next-token distribution and q ∈ ∆|V| denote the draft head’s distribution, where V is the vocabulary. We define the

target entropy as:

H(p) = − ∑

p(v) log p(v), (4)

v∈V

which measures the uncertainty of the target model’s prediction. A low entropy indicates a confident, peaked distribution, while a high entropy indicates a spread-out distribution.

We are interested in understanding how H(p) constrains the achievable acceptance rate αTO and αRS defined in Eq. (1) and (2).

###### 3.2 MTP with Target-Only Sampling

Under target-only sampling, the acceptance rate depends on how well the draft’s greedy prediction yˆ = argmaxy q(y) aligns with the target’s high-probability region. When the target entropy H(p) is low (i.e., p is peaked on a few tokens), even a moderately accurate draft model can achieve high acceptance by placing mass on the dominant tokens. Conversely, when H(p) is high, the target distribution spreads over many tokens, reducing maxy p(y) and increasing the probability of ranking errors.

- Proposition 1 (Entropy-Dependent Acceptance under Target-Only Sampling). For a well-trained draft

model, αTO = maxy p(y), which is a monotonically decreasing function of H(p), lower-bounded by exp(−H(p)), and empirically well-approximated as linear (Fig. 1a):

αTO ≈ aTO − bTO · H(p), (5)

with positive constants aTO, bTO. Ranking errors under imperfect drafts steepen the slope but preserve linearity (§D.2).

Proof sketch. When the draft correctly identifies the target’s top-1 token (argmax q = argmax p), the acceptance rate reduces to αTO = maxy p(y). By Jensen’s inequality applied to the concave logarithm, log(maxy p(y)) ≥ −H(p), i.e., maxy p(y) ≥ exp(−H(p)). Writing αTO = f(H) for some smooth decreasing function f and performing a first-order Taylor expansion around a reference entropy H¯:

αTO ≈ f(H¯ ) − f′(H¯ )H¯

aTO

+ f′(H¯ )

−bTO

·H(p). (6)

Since f is decreasing, bTO = −f′(H¯ ) > 0. See §D.2 for the full derivation including imperfect draft corrections.

| |
|---|

This linear relationship is remarkably robust across different model sizes, tasks, and training stages, as shown in Fig. 1a.

3.3 MTP with Rejection Sampling

Under rejection sampling, the acceptance rate equals the TV overlap between p and q (Eq. (2)). We can decompose the TV distance using the identity |a − b| = a + b − 2min(a, b) and probability normalization:

dTV(p, q) =

- 1

- 2 ∑

v

p(v) +q(v)−2min(p(v),q(v)) = 1−∑

v

min p(v), q(v) . (7)

Therefore, maximizing the acceptance rate is equivalent to minimizing the TV distance:

αRS = 1 − dTV(p, q). (8)

As a result, the acceptance rate is no longer bounded by the policy’s entropy directly. However, empirical results show that the connection to entropy remains after switching to rejection sampling. In our further investigation, we find that under CE/KL-trained draft models, even small per-token mismatches accumulate when p has high entropy, leading to a larger TV distance. This motivates our deeper analysis of how the training objective affects this relationship as follows.

- Proposition 2 (Entropy-Dependent Acceptance under CE/KL-Trained Rejection Sampling). Under CE/KLtrained draft models, the rejection sampling acceptance rate satisfies:

αRS ≈ aRS − bRS · H(p), (9) with positive constants aRS, bRS, where bRS is comparable to bTO though empirically slightly steeper (§D.3, Fig. 8).

Proof sketch. The CE/KL gradient qj − pj produces uniform per-token mismatch |ηv| ≲ σ. Since the effective support size scales as |Seff| ≈ exp(H(p)), the TV distance accumulates as dTV ≈ σ2 exp(H(p)), yielding αRS ≈ 1 − σ2 exp(H(p)). Linearizing the exponential over the operating entropy range gives the stated form. See §D.3 for details.

| |
|---|

Therefore, under CE/KL-trained MTP, both rejection and target-only sampling remain sensitive to entropy shifts. As policy entropy fluctuates significantly during RL training, this sensitivity inherently limits the achievable speedup.

#### 4 Optimizing MTP for RL Training

As discussed above, MTP acceptance rates can degrade significantly during RL training due to the entropy bound. In this section, we develop the novel end-to-end TV loss to address this challenge.

###### 4.1 TV Loss: Directly Optimizing Acceptance Rate

Motivation. Conventional MTP training minimizes the cross-entropy (CE) loss or the KL divergence between the target and draft distributions.4 However, the rejection sampling acceptance rate is determined by the TV distance (Eq. (8)), not the KL divergence. By Pinsker’s inequality, dTV(p, q) ≤ DKL(p∥q)/2, so KL provides only an indirect upper bound, and minimizing it does not efficiently minimize TV distance. This motivates directly optimizing the TV distance as the MTP training objective.

TV Loss. We propose to directly minimize the TV distance:

LTV = dTV(p,q) = 1− ∑

min p(v), q(v) , (10)

v∈V

where p is treated as a constant (detached from the computation graph) and gradients flow only through q.

Gradient Analysis. Let the draft head output logits z ∈ R|V| with qj = softmax(z)j. The gradient of the TV loss with respect to zj is:

∂LTV ∂zj

= −qj [q j ≤ pj]−S , where S = ∑

[q v ≤ pv] · qv. (11)

v

- Proposition 3 (Bounded Gradient). The TV loss gradient is bounded: ∂L∂zTV

≤ 1 for all j.

j

Proof. Since qj ∈ [0,1] and |[q j ≤ pj] − S| ≤ 1 (as both the indicator and S ∈ [0,1]), we have ∂L∂zTV

###### =

j

qj · |[q j ≤ pj] − S| ≤ 1. This bounded gradient property ensures training stability, in contrast to KL divergence whose gradient ∂DKL

| |
|---|

∂zj = qj − pj can exhibit large magnitudes when q and p disagree significantly.

Intuitive Interpretation. The TV loss gradient has a natural interpretation in terms of the rejection sampling mechanism:

- • For tokens where qj ≤ pj (tokens that would be accepted): the gradient increases the logit, encouraging the draft to assign more mass.
- • For tokens where qj > pj (tokens that would be rejected): the gradient decreases the logit, suppressing overconfident predictions.
- • For tokens where qj ≈ 0 (irrelevant tokens): the gradient is automatically ≈ 0 (since it is proportional to qj), avoiding wasted optimization effort on the long tail of the vocabulary.

This selective gradient behavior contrasts with KL divergence, which applies gradients to all tokens regardless of their relevance to the acceptance decision.

4Throughout this paper, “KL” refers to the forward KL divergence DKL(p∥q) unless otherwise noted. CE and forward KL differ only by the constant H(p) and yield identical gradients. We analyze the reverse KL divergence DKL(q∥p) separately in §C.

- Table 1: Gradient comparison across training objectives. C denotes a global constant (S for TV, DKL(q∥p) for reverse KL). See §A-§C for derivations.

Property Forward KL Reverse KL TV Loss Gradient qj − pj qj[log(qj/pj) − C] −qj[[q j ≤ pj] − C]

∝ qj? No Yes Yes Tail suppression No Yes Yes

Comparison of CE, KL, and TV Gradients. Table 1 summarizes the gradient structures of the three training objectives. The key distinction lies in whether the gradient is proportional to qj: CE loss produces uniform per-token mismatch (qj − pj) that distributes optimization effort uniformly across the vocabulary, including irrelevant low-probability tokens. In contrast, both reverse KL and TV loss exhibit qj-proportional gradients with natural tail suppression, concentrating updates on tokens the draft already assigns non-negligible mass. However, despite this shared property, reverse KL yields negligible acceptance rate improvements over CE (§6), because its zero-forcing behavior allows the draft to drop modes of p and its asymmetric penalty drives q ≤ p globally—both reducing the TV overlap ∑v min(p, q) (see §C for a detailed analysis). TV loss avoids these pitfalls by directly optimizing the acceptance-relevant quantity and producing a probability-proportional mismatch that decouples acceptance from target entropy.

- 4.2 End-to-End Multi-Step TV Loss For γ-step MTP, the expected acceptance length is:

E[L] =

γ

∑

j=1

j

∏

i=1

αi = α1 + α1α2 + α1α2α3 + · · · +

γ

∏

i=1

αi, (12)

where αi = 1 − dTV(pi, qi) is the per-step acceptance rate at step i. Directly optimizing the average per-step TV distance γ1 ∑iγ=1 dTV(pi, qi) does not account for the multiplicative structure of multi-step acceptance. We therefore propose the end-to-end (e2e) TV loss:

Le2e = 1 −

1 γ

γ

∑

j=1

j

∏

i=1

αi = 1 −

1 γ

γ

∑

j=1

j

∏

i=1

1 − dTV(pi, qi) . (13)

This loss directly optimizes the normalized expected acceptance length, naturally weighting earlier steps more heavily (since they appear in more product terms) and capturing the compounding effect of multi-step verification. This can be regarded as a dynamic step-wise weighting scheme: since αi depends on the current draft quality, the effective weight of each position adapts automatically as training progresses, shifting emphasis toward steps that currently limit acceptance. This contrasts with prior work that uses fixed position-dependent weights, such as head-dependent loss weights (Cai et al., 2024; Li et al., 2026), exponentially decaying block-position weights (Chen et al., 2026a), fixed decay on rejected positions (Lei et al., 2026), or per-position weights on a CE base (Wu et al., 2026).

- 4.3 Impact of Training Objective on Entropy-Acceptance Relationship

Having introduced the TV loss, we now analyze why it fundamentally outperforms CE/KL training in the context of RL, where the target entropy shifts continuously. The linear relationships in Eq. (5) and (9) characterize draft models trained with CE/KL loss; we show that the choice of training objective fundamentally alters the entropy-acceptance relationship. The full derivation is provided in §D; here we state the main results.

###### Pinsker’s inequality and the KL–TV gap. By Pinsker’s inequality:

- 1

- 2

DKL(p∥q), (14)

dTV(p, q) ≤

√DKL/2 provides only an upper bound on dTV, and KL optimization allocates model capacity inefficiently for minimizing TV distance: Minimizing the KL divergence does not efficiently minimize the TV distance, which is the quantity that directly determines the rejection sampling acceptance rate.

###### CE/KL Training: Uniform Mismatch. The KL divergence gradient ∂D∂zKL

= qj − pj applies optimization pressure proportional to the absolute difference |qj − pj|, regardless of the magnitude of pj relative to other tokens. Under a capacity-limited draft model, this produces approximately uniform per-token mismatch: |q∗(v) − p(v)| ≲ σ for a constant σ. As shown in Proposition 2, this uniform mismatch accumulates over the effective support |Seff| ≈ exp(H(p)), yielding an entropy-dependent acceptance rate.

j

TV Training: Probability-Proportional Mismatch. The TV loss gradient (Eq. (11)) is proportional to qj, concentrating optimization on high-probability tokens and automatically ignoring the long tail. Under a capacity-limited draft model, each token receives optimization resources proportional to its probability qj ≈ pj, so the per-token mismatch also scales with p(v) rather than remaining at a uniform level. This produces probability-proportional mismatch: |q∗(v) − p(v)| ≲ δ · p(v) for a constant δ (see §D.4 for a detailed derivation).

- Proposition 4 (Reduced Entropy Dependence under TV Training). When the per-token mismatch satisfies |q∗(v) − p(v)| ≲ δ · p(v), the TV distance is bounded independently of entropy:

dTV(p, qTV∗ ) ≤

δ

2 ∑

v

p(v) =

δ 2

, (15)

yielding αRSTV ≥ 1 − δ/2. In practice, the draft head has finite capacity, so δ may exhibit weak entropy dependence δ = δ(H), but empirically the entropy–acceptance slope is reduced by over 95% compared to CE/KL training (Fig. 8).

Proof sketch. The TV gradient is proportional to qj (Eq. (11)), so each token’s optimization resource scales with its probability, producing |q∗(v) − p(v)| ≲ δ · p(v) (§D.4). Summing: dTV = 12 ∑v |q∗ − p| ≤ δ

2 ∑v p(v) = 2δ, which is entropy-independent since ∑v p(v) = 1.

| |
|---|

This analysis explains the empirical observation that TV-trained draft models achieve substantially more stable acceptance rates across varying target entropy, while CE/KL-trained models exhibit a strong negative correlation (Fig. 8).

- 5 MTP Adaptation Strategy for RL

A key question for using MTP in RL pipelines is whether we need online updates of the MTP module during RL training. We investigate this through a decomposition analysis that disentangles the two factors driving acceptance rate changes.

###### 5.1 Decomposition: Entropy vs. Mismatch in RL

Using the linear entropy–acceptance relationship established in §3, we decompose the change in acceptance length during RL training as:

###### ∆αt = b · (Ht − H0)

###### + ∆αt − b · (Ht − H0)

, (16)

∆αentropy

∆αmismatch

where b is the entropy–acceptance slope estimated from the early phase of each experiment, H0 is the initial entropy, and ∆αt = αt − α0 is the total acceptance change at step t. The first term captures the acceptance change attributable to entropy shifts alone (assuming a fixed draft–target relationship), while the residual captures the effect of growing draft–target mismatch due to backbone weight updates.

As shown in Fig. 3: (1) Under target-only sampling, both entropy increase and growing mismatch contribute to acceptance degradation, as the greedy draft prediction becomes increasingly misaligned with the evolving target. (2) Under rejection sampling with CE loss, the degradation is almost entirely entropy-driven (∆αmismatch ≈ 0), indicating that RL weight updates do not significantly affect the draft– target TV overlap. (3) Under rejection sampling with TV loss, near-zero change is observed across all components, confirming that TV-trained drafts are robust to both entropy shifts and weight updates.

###### 5.2 Pre-RL Adaptation is Sufficient

The decomposition analysis leads to a key practical insight: since the draft–target mismatch induced by RL weight updates is negligible under rejection sampling, updating the MTP heads during RL is unnecessary.

###### Target Only

###### RS w/ CE Loss

###### RS w/ TV Loss

0.10

0.10

0.10

0.05

0.05

0.05

0.00

0.00

0.00

0.05

0.05

0.05

AcceptLength

0.10

0.10

0.10

0.15

0.15

0.15

0.20

0.20

0.20

(total)

(total)

(total)

0.25

0.25

0.25

entropy

entropy

entropy

mismatch

mismatch

mismatch

0.30

0.30

0.30

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

RL Training Progress (normalized)

- Figure 3: Decomposition of acceptance length changes during RL training. ∆α (total, gray) is decomposed into an entropy-driven component ∆αentropy = b · (Ht − H0) (orange) and a draft–target mismatch

component ∆αmismatch (green). Under target-only sampling, both entropy increase and growing mismatch contribute to acceptance degradation. Under rejection sampling with CE loss, the degradation is almost entirely entropy-driven, with mismatch remaining near zero. RS with TV loss shows near-zero change across all components, confirming the stability of TV-trained drafts.

A one-time pre-RL adaptation with TV loss—applied during the SFT stage before RL begins—is sufficient to produce draft models that maintain high acceptance rates throughout RL training (Fig. 6). This eliminates the memory overhead of maintaining MTP optimizer states and the computational cost of MTP gradient updates during RL.

Empirically, as shown in Fig. 9a, continuing to update MTP weights during RL yields no significant improvement when starting from a well-trained TV checkpoint. Worse, updating with CE loss during RL causes the acceptance rate to degrade toward the RS w/ CE baseline, as CE loss makes the draft distribution smoother and erodes the gains from TV training (§7.2).

###### 5.3 Cross Training of MTP and Backbone

When MTP co-training during RL is desired (e.g., for target-only sampling where mismatch is nonnegligible), we find that joint training with separate learning rates and separate gradient norm normalization provides the best trade-off. The backbone gradients are not affected by the MTP loss (which only flows through the draft heads), ensuring that the MTP training does not interfere with the RL optimization of the backbone.

#### 6 Experiments

We validate the effectiveness of our method Bebop through three sets of experiments: (1) the impact of different multi-step MTP loss objectives on acceptance rate during SFT; (2) the benefits of e2e TV loss with rejection sampling on acceptance rate, speedup, and training stability during RL; and (3) the gains from updating MTP parameters during the RL stage.

###### 6.1 Multi-Step MTP Training Improves Acceptance Rate

We first evaluate how different loss objectives affect MTP acceptance rates during the SFT stage. Specifically, we compare four MTP training objectives:

- (1) CE loss: standard cross-entropy between draft and target distributions;
- (2) KL loss: KL divergence DKL(p∥q);
- (3) Reverse KL loss: Reverse KL divergence DKL(q∥p) (Eq. (17));
- (4) TV loss: per-step TV distance (Eq. (10));
- (5) e2e TV loss: end-to-end multi-step TV loss (Eq. (13)).

We conduct the primary experiments on Qwen3.5-35A3B (Qwen Team, 2026a) using mixed RFT data. All experiments use a constant learning rate of 3.5 × 10−5 with 3% warmup steps, training for 1 epoch with Megatron (Shoeybi et al., 2019) at a global batch size of 256 and a sequence length of 256K. During multi-step MTP training, we perform forward and backward passes over 5 MTP steps while freezing the

- Table 2: MTP acceptance rate (%) under rejection sampling across tasks and training objectives under γ = 3 on Qwen3.5-35A3B. All results are measured at convergence. ∆ denotes improvement over CE loss baseline.

###### MTP Loss Math Code SWE Agent MTBench (OOD)

CE loss (baseline) 75.0 71.3 75.1 90.3 65.3 KL loss +0.0 +0.0 +0.2 +0.2 +0.0 Reverse KL loss +1.3 +1.0 −0.2 +1.0 +0.5 TV loss +2.4 +2.5 +3.3 +5.2 +1.4 e2e TV loss (ours) +3.0 +3.3 +8.0 +6.7 +2.3

LLM backbone. All evaluations use γ = 3 (i.e., the target model verifies 4 tokens at a time). We further extend our experiments to Qwen3.6-35A3B, Qwen3.6-Plus, and Qwen3.7-Plus, training on different data mixtures including domain-specific data (code, agent, reasoning) and mixed RFT data. The throughput is measured using SGLang’s MTP implementation with rejection sampling (see §G for implementation details).

Rejection Sampling Acceptance. Table 2 reports the acceptance rate improvements of our proposed e2e TV loss compared to the CE and KL baselines on Qwen3.5-35A3B. Across all tasks, e2e TV loss consistently improves rejection sampling acceptance rates by 3–8% on in-distribution tasks (Math, Code, Agent, SWE) and up to 2.3% on the out-of-distribution MT-Bench (Zheng et al., 2023) task. Notably, on Agent tasks where the CE baseline already achieves a high acceptance rate of 90.3%, e2e TV loss further pushes it to 97.0%, a level that substantially improves rollout efficiency in both RL training and agentic inference.

Beyond the primary experiments, we evaluate across a broader set of models and data configurations. As shown in Fig. 4, we train Qwen3.6-35A3B, Qwen3.6-Plus, and Qwen3.7-Plus on different data mixtures and track per-step acceptance rates throughout training. Several patterns emerge. First, CE loss causes a pronounced and persistent decline in Step 1 acceptance rate during training, as it distributes optimization effort across the entire vocabulary. In contrast, TV loss maintains stable or slightly improving Step 1 acceptance. Second, the advantage of e2e TV loss becomes increasingly prominent at later MTP steps: at Step 3, TV loss outperforms CE loss by approximately 5%, while at Step 2 the margin is 2.5–5%. Third, the gains are task-dependent: agentic tasks benefit the most, with improvements up to 8% on Agent and SWE-Bench (Jimenez et al., 2024), while reasoning and conversational tasks see gains of 4–5%. Finally, MTP acceptance rates exhibit strong generalization. Models trained entirely without agent-specific data still achieve approximately 70% acceptance on agent tasks. Specifically, TV loss provides larger improvements on in-distribution domains than on out-of-distribution tasks.

Target-Only Acceptance. Under target-only sampling, acceptance rates are nearly identical across all training objectives (<0.3% difference), as shown in Fig. 5. This is expected: target-only acceptance

αTO = p(argmaxy q(y)) reduces to maxy p(y) when the draft’s top-1 ranking is correct, depending only on the target distribution rather than the draft’s distributional shape. In contrast, rejection sampling

acceptance αRS = ∑v min(p(v), q(v)) depends on the full distributional overlap, which is where TV loss provides its advantage. This is consistent with our analysis in §3.

Throughput. As shown in Fig. 9b, the acceptance rate improvement translates to throughput gains roughly linearly. The e2e-TV-trained Qwen3.7 Plus consistently outperforms the CE-loss-trained Qwen3.6 Plus on all datasets. These gains effectively accelerate RL rollouts, which is significant at the scale of hundreds of thousands of GPU hours.

Acceptance Rate Scales with Model Size. As shown in Table 3, MTP acceptance rates after multi-step SFT training consistently increase with model size. Qwen3.7 models are trained with e2e TV loss, while Qwen3.6 models use CE loss. The acceptance rate reaches up to 95%, especially on agent tasks, indicating that the draft model under γ = 3 has nearly converged to the backbone model. Conversely, as model size decreases, acceptance rates degrade to varying degrees.

###### 6.2 TV Loss Stabilizes MTP Acceleration in RL Training

We conduct extensive experiments in RL settings to demonstrate the effectiveness of Bebop. We select two representative workloads spanning different generation regimes:

###### MTP Step 1

###### MTP Step 2

###### MTP Step 3

###### Accept Length

CE TV

CE TV

CE TV

0.90

0.775

0.65

3.3

0.750

0.60

0.89

3.2

AcceptanceRate

0.725

0.55

0.88

0.700

3.1

Code

0.50

0.87

0.675

0.45

3.0

0.86

0.650

0.40

2.9

0.625

0.85

0.35

CE TV

2.8

0.600

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.825 CE TV

CE TV

CE TV

0.92

0.70

0.800

3.4

0.65

0.775

0.91

3.3

AcceptanceRate

0.60

AcceptLength

0.750

Math

0.55

3.2

0.90

0.725

0.50

3.1

0.700

0.89

0.45

0.675

3.0

0.40

0.88

CE TV

0.650

2.9

0.35

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.835

CE TV

CE TV

CE TV

0.50

3.00

0.64

0.830

2.95

0.45

0.62

0.825

AcceptanceRate

2.90

0.820

0.60

MT-Bench

0.40

2.85

0.815

0.58

2.80

0.810

0.35

0.56

2.75

0.805

0.54

0.30

2.70

0.800

CE TV

0.52

0.795

2.65

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

Normalized Training Progress

Normalized Training Progress

Normalized Training Progress

(a) Accept length on reasoning and conversation tasks (Math, Code, MT-Bench).

###### Hybrid

###### Agent

###### Long-Horizon

###### SWE-Bench

CE TV

CE TV

CE TV

CE TV

3.2

3.8

3.1

3.6

3.1

3.6

3.0

AcceptLength

AcceptLength

AcceptLength

AcceptLength

3.5

3.0

3.4

2.9

2.9

3.2

3.4

2.8

3.0

2.8

3.3

2.8

2.7

2.7

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

Normalized Training Progress

Normalized Training Progress

Normalized Training Progress

(b) Accept length on agentic and hybrid tasks (Hybrid, Agent, Long-Horizon, SWE-Bench).

- Figure 4: CE loss (solid) vs. TV loss (dashed) during SFT training. TV loss consistently achieves higher acceptance rates across all MTP steps, with especially pronounced gains on agentic tasks.

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

2.87

2.88

2.89

2.90

2.91

2.92

2.93

2.94

AcceptLength

Hybrid

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

2.70

2.71

2.72

2.73

2.74

2.75

2.76

2.77

AcceptLength

MT-Bench

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

3.18

3.20

3.22

3.24

3.26

AcceptLength

SWE

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Training Progress

2.66

2.68

2.70

2.72

2.74

2.76

2.78

2.80

AcceptLength

Long-Horizon

CE TV

- Figure 5: Accept length under target-only sampling with CE loss vs. TV loss during SFT training. Acceptance rates are nearly identical (<0.3% difference) across all tasks, confirming that target-only acceptance depends on the target distribution rather than the draft’s distributional shape.

- (1) Reasoning RL: long chain-of-thought tasks including math reasoning, code reasoning, and instructionfollowing, with a maximum generation length of 64K tokens. Evaluation benchmarks: HMMT25 (Dekoninck et al., 2026), AIME25 (Zhang and Math-AI, 2025), and LiveCodeBench (Jain et al., 2025).
- (2) SWE RL: multi-turn code editing tasks where each turn involves thinking, tool calling, and tool execution, with tool responses appended to the previous context. Maximum generation length is 128K tokens with up to 200 turns. Evaluation benchmark: SWE-Verified (Jimenez et al., 2024).

For all RL experiments, we use SGLang (Zheng et al., 2024) as the rollout engine within an asynchronous RL framework built on top of veRL (Sheng et al., 2024), with a learning rate of 1 × 10−6 or 2 × 10−6.

- Fig. 6 shows the accept length trends during RL training. With rejection sampling and TV loss, Bebop

3.6

3.3

3.70

3.5

3.65

3.2

3.60

3.4

AcceptLength

AcceptLength

AcceptLength

3.1

3.55

RS w/ TV

3.3

RS TO

3.50

3.2

3.0

3.45

3.1

2.9

3.40

RS w/ TV RS w/ CE TO

3.0

RS w/ TV

3.35

TO

2.8

0 50 100 150 200 250 300

0 25 50 75 100 125 150 175 200

0 20 40 60 80 100 120

Training Step

Training Step

Training Step

(a) Reasoning RL.

(b) SWE RL.

(c) SWE RL in Qwen-3.7 Max.

- Figure 6: Accept length during RL training across different workloads in Qwen3.6-Plus and Qwen3.7-Max. Rejection sampling with TV loss (RS w/ TV) consistently maintains higher accept lengths compared to target-only (TO) and rejection sampling with CE loss (RS w/ CE).

- Table 3: MTP acceptance rate (%) under rejection sampling across tasks and training objectives under γ = 3 on different models. Qwen3.7 models are trained with e2e TV loss; all others are trained with CE loss.

###### Model Math Code Hybrid SWE Agent Long-horizon MTBench

Qwen3.7-Max 87.6 87.7 78.1 81.9 94.6 77.2 73.2 Qwen3.7-Plus 87.4 85.7 75.3 79.2 98.6 78.0 74.3 Qwen3.6-Plus 82.2 78.7 72.2 75.2 99.1 75.6 71.0 Qwen3.6-27B 79.9 76.7 71.9 72.3 96.3 69.5 67.5 Qwen3.6-35A3B 78.3 74.4 69.2 71.3 97.1 71.3 65.2

maintains stable or improving acceptance length throughout training, even as the policy maintains high entropy. In Reasoning RL, the observed increase in acceptance rate is primarily driven by a significant drop in policy entropy during training, rather than improved draft alignment alone. In contrast, the SWE workloads exhibit slightly increasing entropy, making them a more direct test of the training objective’s robustness: here, RS w/ TV maintains stable accept lengths while target-only sampling suffers continuous degradation. The advantage is most pronounced on SWE and other high-entropy tasks, where higher accept lengths translate directly into faster rollout completion. Furthermore, at larger model scales (Fig. 6c), RS w/ TV exhibits a stronger entropy-invariant trend and sustains high acceptance rates throughout RL training, whereas target-only sampling shows a persistent acceptance rate decline.

- Fig. 7 shows the corresponding latency improvements. MTP with rejection sampling achieves 1.5–1.8× reduction in per-step RL training latency compared to training without MTP, with the largest gains on agentic tasks where the rollout phase achieves up to 2.4× speedup in Agentic RL. These speedups are consistent across all workloads and provide substantial wall-clock savings at scale.

Fig. 1a and Fig. 8 validate the linear entropy–acceptance relationships established in §3. Notably, training with TV loss substantially reduces the entropy–acceptance slope (by over 95%, e.g., from −1.68 to −0.06) and shifts the intercept upward. This confirms that TV loss improves acceptance both by better aligning the draft distribution with the target and by largely decoupling the acceptance rate from the target entropy, consistent with the entropy-invariant mismatch structure analyzed in §4.3, thereby enabling stable MTP acceleration gains throughout RL training.

###### 6.3 Benefits of Updating MTP Weights During RL

After thorough multi-step SFT training, the model already achieves high acceptance rates (e.g., above 75% for Qwen3.7-Max). As long as the acceptance rate is maintained, the MTP acceleration benefits are preserved throughout RL training. Furthermore, the analysis in §4 and the experimental validation in Fig. 1a demonstrate that rejection sampling with TV loss effectively decouples the entropy–acceptance relationship, stabilizing acceptance rates during RL. To further quantify the benefits of updating MTP weights during RL, we compare the following training configurations:

- (1) RS w/ TV + TV loss: starting from the RS w/ TV checkpoint and online MTP training with TV loss;
- (2) RS w/ TV + CE loss: starting from the RS w/ TV checkpoint and online MTP training with CE loss;
- (3) TO + CE loss: starting from the TO checkpoint and continuing MTP training with CE loss.

###### SWE Rollout Avg Latency

###### SWE Step Latency

###### Latency During RL Training

1400

4000

RS w/ TV

RS w/ TV

RS w/ TV

w/o MTP

w/o MTP

TO

w/o MTP

3500

1200

3000

3000

Latency(s/step)

Latency(s/step)

RolloutAvg(s)

1000

2500

2500

800

2000

2000

600

1500

1500

400

1000

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 20 40 60 80 100 120 140

Training Step

Training Step

Training Step

(a) Reasoning RL.

(b) SWE RL.

###### Agent Rollout Avg Latency

###### Agent Step Latency

8000

RS w/ TV

RS w/ TV

w/o MTP

w/o MTP

250

7000

6000

200

Latency(s/step)

RolloutAvg(s)

5000

150

4000

100

3000

2000

50

1000

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Step

Training Step

(c) Agent RL.

- Figure 7: Training latency comparison during RL using Qwen3.6-35A3B and Qwen3.6-Plus. MTP with rejection sampling (RS w/ TV) substantially reduces per-step latency compared to training without MTP (w/o MTP) and target-only sampling (TO).

0.34 0.36 0.38 0.40 0.42 0.44

Entropy Loss

2.8

2.9

3.0

3.1

3.2

3.3

AcceptLength

RS w/ TV RS w/ CE TO

(a) Reasoning RL

0.185 0.190 0.195 0.200 0.205 0.210 0.215 0.220 0.225

Entropy Loss

3.3

3.4

3.5

3.6

3.7

AcceptLength

RS w/ TV RS w/ CE TO

(b) SWE RL

0.39 0.40 0.41 0.42 0.43 0.44 0.45

Entropy Loss

2.9

3.0

3.1

3.2

3.3

3.4

3.5

3.6

AcceptLength

RS w/ TV

TO

(c) SWE RL in Qwen-3.7 Max

- Figure 8: Entropy loss vs. accept length across three RL workloads in Qwen3.6-Plus and Qwen3.7-Max. Each point represents one training step; the line shows the linear fit. TO and RS w/ CE exhibit a strong negative correlation (slope ≈ −1.68), while RS w/ TV remains nearly flat (slope ≈ −0.06), confirming that TV training decouples acceptance from entropy.

As shown in Fig. 9a, as RL training with MTP weight updates progresses, the acceptance rate converges toward the corresponding baseline without weight updates. For example, although RS w/ TV initially achieves a higher acceptance rate due to TV loss training, updating the MTP weights with CE loss during RL causes the acceptance rate to degrade toward that of RS w/ CE. This shift in acceptance rate reflects changes in the draft distribution: as analyzed in §7.2, CE loss updates make the RS w/ TV draft distribution smoother, bringing it closer to the RS w/ CE distribution. Moreover, for already well-trained MTP weights, further parameter updates during RL yield no significant improvement, with the acceptance rate closely tracking the non-updated baseline. In the case of target-only sampling, updating with CE loss can even cause acceptance rate degradation due to distribution mismatch between the draft and target models.

#### 7 Discussion

In this section, we provide a deeper analysis of the mechanisms behind e2e TV loss and rejection sampling, including the distributional effects of TV loss, comparison of the robustness of different acceptance methods, and analysis of how temperature, generation length, and agentic workloads affect

3.3

3.2

AcceptLength

3.1

3.0

RS w/ TV RS w/ CE TO

2.9

RS w/ TV + MTP Train w/ TV RS w/ TV + MTP Train w/ CE TO + MTP Train w/ CE

2.8

0 25 50 75 100 125 150 175 200

Training Step

(a) Accept length with MTP weight updates.

Math

Qwen3.6-27B Qwen3.6-Turbo

| |
|---|

GLM-5

- Qwen3.5-Plus

Nemotron-Super

MiMo-v2.5

- Qwen3.6-Plus

- Qwen3.7-Plus

1.3

ThroughputRatio(RS/No-RS)

MT-Bench

1.2

MT-Bench Math

Code

Code

Math

1.1

Math MT-Bench

MT-Bench

Math

Code

Code

Code

MathCode

Code Math

| |
|---|

Code

| |
|---|

1.0

MT-Bench

Math

MT-Bench

MT-Bench

0.00 0.05 0.10 0.15 0.20 Accept Rate Delta (RS No-RS)

(b) Accept rate delta vs. throughput ratio.

- Figure 9: (a) Accept length during RL training with and without MTP weight updates. Updating MTP weights with CE loss causes the acceptance rate to converge toward the corresponding nonupdated baseline, while target-only sampling with CE loss updates can even degrade acceptance due to distribution mismatch. (b) Accept rate delta (RS − No-RS) vs. throughput speedup ratio (RS / No-RS) across 8 models and 3 tasks (r = 0.81). Higher acceptance rate gains from rejection sampling translate directly to greater throughput improvements.

- 6

- 7

- 8 MT-Bench

MT-Bench

-0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8

Entropy Gap H = H(q) H(p)

- 0

- 1

- 2

- 3

Code Math

MT-Bench

Code

Math

MT-Bench

Code

Math

Code Math

MT-Bench

Code Math MT-Bench Code

Math

Code

Math

MT-Bench

Code

Math

MT-Bench

Qwen3.6-27B

Qwen3.6-35A3B

GLM-5

- Qwen3.5-Plus

Nemotron-Super

MiMo-v2.5

- Qwen3.6-Plus

- Qwen3.7-Plus

()DqpKL

(a) Entropy gap vs. KL divergence.

-0.4 -0.2 0.0 0.2 0.4 0.6 Entropy Gap H = H(q) H(p)

0.5

0.6

0.7

0.8

0.9

1.0

RSAcceptRate

| |Math|
|---|---|
| | |

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Qwen3.6-27B Qwen3.6-35A3B

| |
|---|

GLM-5

- Qwen3.5-Plus

Nemotron-Super

MiMo-v2.5

- Qwen3.6-Plus

- Qwen3.7-Plus

slope=-0.48, r=-0.16

(b) Entropy gap vs. RS accept rate.

0 1 2 3 4 5 6 7 DKL(q p)

0.60

0.65

0.70

0.75

0.80

0.85

RSAcceptRate

| |Math|
|---|---|
| | |

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Qwen3.6-27B Qwen3.6-35A3B

| |
|---|

GLM-5

- Qwen3.5-Plus

Nemotron-Super

MiMo-v2.5

- Qwen3.6-Plus

- Qwen3.7-Plus

slope=-0.00, r=-0.32

(c) KL divergence vs. RS accept rate.

- Figure 10: (a) Entropy gap ∆H vs. DKL(q∥p) across models and tasks. (b) Entropy gap correlates negatively with RS acceptance rate (r = −0.54). (c) KL divergence shows no such correlation (r = 0.13), indicating that entropy gap, rather than KL, is the relevant predictor of RS acceptance.

MTP acceptance.

###### 7.1 TV Loss Makes Draft Distributions Sharper

We analyze how the TV loss affects the draft distribution’s entropy compared to CE/KL training. The TV loss produces draft distributions with entropy closer to the target entropy (but slightly higher), indicating that the draft becomes sharper and more aligned with the target’s peaked predictions. In contrast, CE/KL training tends to produce smoother draft distributions that spread mass across the vocabulary, which is suboptimal for rejection sampling where the overlap ∑v min(p(v), q(v)) is maximized by matching the target’s shape.

This sharpening effect arises from the TV loss gradient’s selective behavior (Eq. (11)): it focuses optimization effort on tokens near the decision boundary (qj ≈ pj) while ignoring irrelevant low-probability tokens. Fig. 10 illustrates the relationship between the draft–target entropy gap and KL distance across models. Models with well-trained MTP heads exhibit a smaller entropy gap between draft and target distributions, while having a larger KL distance (see also Fig. 1b).

###### 7.2 Different MTP Training Losses Induce Different Draft Distribution Patterns

- Fig. 11 shows how various MTP metrics evolve when updating MTP weights with different losses during RL. TV loss produces draft entropy closer to the target model, but with a larger KL distance compared to CE loss. Furthermore, because TV loss yields a sharper draft distribution, the corresponding αp>q is lower while αq>p is higher. When different losses are used for MTP weight updates during RL, the MTP

DKL(P Q)

H

p > q

p < q

2.0

0.40

RS w/ TV RS w/ CE

RS w/ TV RS w/ CE

0.65

RS w/ TV + MTP Train w/ TV RS w/ TV + MTP Train w/ CE

RS w/ TV + MTP Train w/ TV RS w/ TV + MTP Train w/ CE

0.35

0.35

1.8

0.60

0.30

0.30

1.6

0.55

0.25

RS w/ TV RS w/ CE

0.25

RS w/ TV + MTP Train w/ TV RS w/ TV + MTP Train w/ CE

0.20

1.4

0.50

0.15

0.20

1.2

0.45

0.10

0.15

RS w/ TV RS w/ CE

1.0

0.05

0.40

RS w/ TV + MTP Train w/ TV RS w/ TV + MTP Train w/ CE

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

Training Step

Training Step

Training Step

Training Step

- Figure 11: Evolution of MTP metrics during RL training with different MTP loss objectives. TV loss

produces draft entropy closer to the target but with larger KL distance, lower αp>q, and higher αq>p. Switching the MTP training loss during RL causes the metrics to shift toward the pattern characteristic of the new loss.

metrics shift toward the pattern characteristic of that loss. For example, with RS w/ TV + CE loss, the draft entropy gradually increases over the course of training.

###### 7.3 Robustness of Acceptance Methods under Policy Updates

Although the analysis in §5.1 shows that the magnitude of model updates during RL is relatively small, an important distinction remains between target-only and rejection sampling in their sensitivity to ranking changes caused by RL policy updates.

Target-only sampling is fragile to ranking shifts. Target-only acceptance relies on whether the draft token falls within the target model’s high-probability region (e.g., top-k). This is a discrete criterion: a token is either accepted or rejected. When an RL gradient step causes the top-1 token to change, even by a small probability shift (e.g., p(v1) drops from 0.31 to 0.29 while p(v2) rises from 0.29 to 0.31), the draft model, still favoring the old top-1, experiences a discontinuous jump from acceptance to rejection.

Rejection sampling degrades smoothly. Under reject sampling, the acceptance rate αRS = ∑v min(p(v), q(v)) is a continuous function of both distributions. The same ranking shift produces a negligible change in the TV overlap, since min(p(v1), q(v1)) + min(p(v2), q(v2)) is nearly invariant to small probability swaps.

High entropy amplifies the fragility gap. When the target entropy is high, multiple tokens have similar probabilities, making ranking changes more frequent under RL updates. This disproportionately affects target-only sampling, where each ranking flip can cause a discrete acceptance failure. Despite this qualitative difference, we empirically observe similar entropy–acceptance slopes for target-only and rejection sampling (bTO ≈ bRS; see §3), suggesting that the discrete fragility of target-only is offset by the cumulative TV distance growth that affects rejection sampling equally under CE/KL training.

###### 7.4 Correlation between Temperature and MTP Acceptance

The sampling temperature τ directly affects the target model’s entropy: H(pτ) = H(softmax(z/τ)) increases monotonically with τ. Combined with the linear entropy-acceptance relationship established in §3, this implies that higher temperatures lead to lower MTP acceptance rates.

- Fig. 12a confirms this: rejection sampling maintains relatively stable acceptance lengths across temperatures, while target-only sampling degrades sharply as temperature increases. This has practical implications for RL training, where higher temperatures are often used to encourage exploration. Our analysis provides a quantitative framework for understanding the throughput cost of exploration via temperature scaling.

7.5 Rejection Sampling Decision Boundary

Rejection sampling outperforms target-only sampling when dTV(p, q) < 1 − p(yˆ), with yˆ=argmaxy q(y) (see §E). This decision boundary provides a simple diagnostic: if the draft–target TVD is smaller than the probability mass outside the draft’s top-1 token under the target, RS is preferred.

- Fig. 13 visualizes this boundary across eight models with natively trained MTP heads, spanning three task categories. Nearly all model–task combinations (23 out of 24) fall firmly in the RS-better region, confirming that for native MTP models, rejection sampling consistently outperforms target-only sampling. This confirms that enabling rejection sampling is beneficial for virtually all practical MTP deployments.

3.50

| |
|---|

3.25

| |
|---|

| |
|---|

| |
|---|

3.00

MeanAcceptLength

2.75

| |
|---|

2.50

2.25

2.00

1.75

RS TO

| |
|---|

1.50

0.1 0.3 0.5 0.7 0.9 1.1 1.3

Temperature

(a) Acceptance length vs. temperature.

0.85

0.80

AcceptRate

0.75

0.70

0.65

RS TO

0.60

0 2 4 6 8 10 12 14 16

Output Length (k)

(b) Accept rate vs. output length.

- Figure 12: (a) Mean acceptance length as a function of sampling temperature. Rejection sampling maintains relatively stable acceptance lengths, while target-only sampling degrades sharply at higher temperatures. (b) MTP acceptance rate vs. output length (averaged over 8 models). RS maintains a stable advantage over target-only sampling across all generation positions.

0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90

1 dTV

0.45

0.50

0.55

0.60

0.65

0.70

0.75

0.80

0.85

()py

RS better

No-RS better Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math MT-Bench

Code

Math MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Code

Math

MT-Bench

Qwen3.6-27B Qwen3.6-35A3B

| |
|---|

GLM-5

Qwen3.5-Plus

Nemotron-Super

MiMo-v2.5

- Qwen3.6-Plus

- Qwen3.7-Plus

RS helped ( >0)

RS hurt ( <0)

- Figure 13: RS decision boundary across models (see §7.5). Nearly all model–task combinations fall in the RS-better region, confirming that rejection sampling is beneficial for virtually all practical MTP deployments.

###### 7.6 Correlation between Generation Length and MTP Acceptance

As shown in Fig. 12b, we observe that MTP acceptance rates vary systematically with the position in the generated sequence. In early positions (close to the prompt), the target model tends to have lower entropy (more predictable continuations), leading to higher acceptance rates. As generation progresses, especially in reasoning tasks with long chains of thought, entropy can increase and acceptance rates may drop. This position-dependent acceptance pattern suggests that adaptive MTP strategies—adjusting the draft length γ based on the estimated local entropy—could further improve throughput.

###### 7.7 Agentic RL and the Bubble Problem

As shown in Fig. 14a, in agentic RL settings (e.g., SWE-bench (Jimenez et al., 2024)), the model generates long, multi-turn interactions that involve tool calls, code execution, and iterative refinement. These settings exhibit particularly long generation lengths and variable entropy profiles, creating periodic fluctuations in acceptance rate that tend to increase as generation progresses.

MTP is especially beneficial in agentic settings for two reasons: (1) long generations contain abundant structured outputs—such as boilerplate code, tool call formats, and repetitive patterns—that are highly predictable, yielding high acceptance rates in these segments; (2) multi-turn interactions and long-tail generation reduce the effective running batch size, a regime where MTP’s latency benefits are amplified

4.00

3.75

3.50

AcceptLength

3.25

3.00

2.75

2.50

Mean

Max

Min

2.25

0 2 4 6 8 10

Time (hours)

(a) Accept length during Agent RL.

###### MTP Loss

full vocab TV

top10k TV top20k TV

0.13

0.12

0.11

0.10

Loss

0.09

0.08

0.07

0.06

0 500 1000 1500 2000 2500 3000

Training Step

(b) MTP loss under top-K truncation.

- Figure 14: (a) Accept length during Agent RL. The mean acceptance length remains stable at ∼3.7, while the min–max range reveals periodic fluctuations across steps. (b) MTP loss curves under different top-K truncation values. Smaller K leads to pronounced loss spikes and training instability, while even K = 20,000 shows slower convergence compared to the full-vocabulary TV loss.

since the inference engine operates further from compute saturation. Indeed, our experiments show that agentic workloads achieve the largest acceptance rate improvements (5%) from our proposed TV loss training.

###### 7.8 Instability of Top-K TV Approximation

Computing the full-vocabulary TV loss incurs high peak memory on large vocabularies. To address this, we employ a fused backward kernel that reduces intermediate activation sizes (see §F). We also experimented with approximating TV loss via a top-K truncation to further reduce peak memory. However, even with K = 20,000, we observe a slight slowdown in loss convergence and performance degradation. Smaller values of K lead to pronounced loss spikes, as shown in Fig. 14b. Ultimately, we adopt the fused full-vocabulary TV loss rather than the top-K approximation.

#### 8 Related Work

Speculative Decoding. Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive LLM inference by using a lightweight draft model to propose multiple tokens, which are then verified by the target model in parallel. Various draft architectures have been proposed, including independent small models (Miao et al., 2024; Shen et al., 2026), early-exit heads (Elhoushi et al., 2024), auxiliary heads (Cai et al., 2024; Li et al., 2024; 2026), MTP heads (DeepSeek-AI, 2024; Gloeckle et al., 2024; Qwen Team, 2026a), and diffusion models (Chen et al., 2026a). Bebop focuses on MTP heads that share the backbone’s hidden states and analyzes their behavior under RL training dynamics.

Reinforcement Learning for LLMs. RL has become central to aligning LLMs with human preferences (Schulman et al., 2017) and enhancing reasoning and agentic capabilities (OpenAI, 2026; DeepSeekAI, 2026; Qwen Team, 2026b). Modern post-training pipelines typically separate RL into rollout, reward evaluation, and policy update stages, while algorithms such as GRPO (Shao et al., 2024) and GSPO (Zheng et al., 2025) improve the optimization objective itself. At the system level, asynchronous or partial-rollout frameworks reduce idle time from long-tail trajectories by decoupling inference workers from training workers (Fu et al., 2025; Wang et al., 2025; THUDM, 2025; Qin et al., 2025). Yet they mainly hide long-tail bubbles, leaving trajectory generation as the bottleneck in long-context, multi-turn, and tool-use settings. Related work studies RL instability from training-inference discrepancy and policy staleness (Yao et al., 2025; Liu et al., 2025); recent MTP methods instead update draft heads online to address draft–target mismatch (Chen et al., 2026b; Iso et al., 2026; MiniMax, 2026b; Li et al., 2025). However, we find that acceptance rate fluctuations during RL are primarily driven by shifts in the target model’s entropy rather than draft–target mismatch, and that target entropy exhibits a linear relationship with MTP acceptance length, an observation also noted by Xiao et al. (2026). Our work is complementary: it accelerates rollout without changing the RL objective or scheduler, and identifies entropy shifts as the dominant factor behind MTP acceptance degradation.

Total Variation Distance in Machine Learning. The TV distance is a standard measure for comparing probability distributions, and has been used in distribution testing (Canonne, 2020), generative modeling (Nowozin et al., 2016), and convergence analysis of Markov chains (Levin and Peres, 2017). In speculative decoding, the rejection-sampling acceptance rate equals the distributional overlap, i.e., α = ∑y min(py, qy) = 1 − dTV(p, q) (Leviathan et al., 2023; Chen et al., 2023). This connection has motivated acceptance-oriented objectives, including LK Losses for directly optimizing speculative decoding acceptance rate (Samarin et al., 2026). However, these works focus on inference-time speculative decoding with a fixed target model. Recent work has also explored using reverse KL to optimize the student model in OPD (Lu and Lab, 2025; Lei et al., 2026), though its training objective still differs substantially from directly maximizing the rejection sampling acceptance rate. To our knowledge, we are the first to propose directly optimizing TV distance as a training objective for MTP heads, and the first to analyze its behavior during RL training.

#### 9 Conclusion

We present Bebop, a systematic study of Multi-Token Prediction (MTP) in the context of reinforcement learning for large language models. Our analysis reveals three key findings: (1) MTP acceptance rates under both target-only and rejection sampling are linearly constrained by the target model’s entropy; (2) Bebop’s end-to-end TV loss directly optimizes multi-step rejection sampling acceptance, yielding ∼10% acceptance-rate improvements, up to 95% acceptance, and up to 25% extra inference throughput over conventional CE/KL objectives; (3) Lightweight pre-RL adaptation with TV loss and rejection sampling is sufficient to maintain high MTP acceptance rates throughout RL training, eliminating the need for costly online MTP updates. Extensive experiments with Qwen3.5, 3.6, and 3.7 models demonstrate that Bebop achieves up to 1.8× end-to-end acceleration in async RL pipelines.

Limitations. Our theoretical analysis of the entropy-acceptance relationship relies on modeling assumptions (uniform vs. probability-proportional mismatch) that are heuristically motivated by gradient structures rather than formally proven; tightening these assumptions remains an open question. Additionally, the entropy invariance guaranteed by TV training is distribution-conditional: it holds within the entropy range covered by the SFT training data, but when RL exploration drives the policy entropy significantly beyond this range, the draft head encounters out-of-distribution target distributions for which the mismatch ratio δ is no longer bounded, restoring an entropy-acceptance dependence comparable to that of CE/KL training. In such cases, MTP co-training with TV loss during RL is recommended to extend the draft head’s effective coverage to the new entropy regime.

#### References

Anthropic. Claude fable 5 and claude mythos 5, 2026. URL https://www.anthropic.com/news/

claude-fable-5-mythos-5.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=PEpbUobfJv.

Cl´ement L. Canonne. A survey on distribution testing: Your data is big. but is it blue? Theory of Computing, 9:1–100, 2020.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. In International Conference on Machine Learning, 2023.

Jian Chen, Yesheng Liang, and Zhijian Liu. Dflash: Block diffusion for flash speculative decoding. ArXiv preprint, abs/2602.06036, 2026a. URL https://arxiv.org/abs/2602.06036.

Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, and Tianwei Zhang. Respec: Towards optimizing speculative decoding in reinforcement learning systems. In Ninth Conference on Machine Learning and Systems, 2026b. URL https://openreview.net/ forum?id=HhDSxs7x2R.

DeepSeek-AI. Deepseek-v3 technical report. ArXiv preprint, abs/2412.19437, 2024. URL https://arxiv.

org/abs/2412.19437.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

Jasper Dekoninck, Nikola Jovanovi´c, Tim Gehrunger, K´ari R¨ognvaldsson, Ivo Petrov, Chenhao Sun, and Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. ArXiv preprint, abs/2605.00674, 2026. URL https://arxiv.org/abs/2605.00674.

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu. LayerSkip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/ 2024.acl-long.681/.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, and Yi Wu. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. ArXiv preprint, abs/2505.24298, 2025. URL https://arxiv. org/abs/2505.24298.

GLM Team. Glm-5.1: Towards long-horizon tasks, 2026. URL https://z.ai/blog/glm-5.1. Accessed: 2026-04-07.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozi`ere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=pEWAcejiU2.

Hayate Iso, Tiyasa Mitra, Sudipta Mondal, Rasoul Shafipour, Venmugil Elango, Terry Kong, Yuki Huang, Seonjin Na, Izzy Putterman, Benjamin Chislett, et al. Accelerating rl post-training rollouts via systemintegrated speculative decoding. ArXiv preprint, abs/2604.26779, 2026. URL https://arxiv.org/abs/ 2604.26779.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=chfJJYC3iL.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.

Kimi Team. Kimi k2.6: Advancing open-source coding, 2026. URL https://www.kimi.com/blog/

kimi-k2-6. Accessed: 2026-04-07.

Haodi Lei, Yafu Li, Haoran Zhang, Shunkai Zhang, Qianjia Cheng, Xiaoye Qu, Ganqu Cui, Bowen Zhou, Ning Ding, Yun Luo, and Yu Cheng. Draft-opd: On-policy distillation for speculative draft models. ArXiv preprint, abs/2605.29343, 2026. URL https://arxiv.org/abs/2605.29343.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 19274–19286. PMLR, 2023. URL https://proceedings.mlr.press/v202/leviathan23a.html.

David A. Levin and Yuval Peres. Markov Chains and Mixing Times. American Mathematical Society, 2 edition, 2017.

Jiajun Li, Yuzhen Zhou, Mao Cheng, and Ruiguo Yang Yang. Power up speculative decoding in reinforcement learning, 2025. URL https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/ main/rlhf/slime/spec/readme-en.md.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: speculative sampling requires rethinking feature uncertainty. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= 1NdN7eXyb4.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=4exx1hUffq.

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, 2025. URL https://richardli.xyz/ rl-collapse.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pages 932–949, 2024.

MiniMax. MiniMax M2.5: Built for real-world productivity. https://www.minimax.io/news/

###### minimax-m25, 2026a.

MiniMax. Forge: Scalable agent rl framework and algorithm. MiniMax News, 2026b. URL https: //www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm. Accessed: 202606-09.

Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using

variational divergence minimization. In NeurIPS, 2016, Barcelona, Spain, pages 271–279, 2016. OpenAI. GPT-5.5 system card, 2026. URL https://openai.com/index/gpt-5-5-system-card/. Ruoyu Qin, Weiran He, Weixiao Huang, Yangkun Zhang, Yikai Zhao, Bo Pang, Xinran Xu, Yingdi

Shan, Yongwei Wu, and Mingxing Zhang. Seer: Online context learning for fast synchronous llm reinforcement learning. ArXiv preprint, abs/2511.14617, 2025. URL https://arxiv.org/abs/2511. 14617.

Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026a. Qwen Team. Qwen3.7: The agent frontier, 2026b. URL https://qwen.ai/blog?id=qwen3.7. Alexander Samarin, Sergei Krutikov, Anton Shevtsov, Sergei Skvortsov, Filipp Fisin, and Alexander

Golubev. Lk losses: Direct acceptance rate optimization for speculative decoding. ArXiv preprint, abs/2602.23881, 2026. URL https://arxiv.org/abs/2602.23881.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv preprint, abs/1707.06347, 2017. URL https://arxiv.org/abs/1707. 06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv preprint, abs/2402.03300, 2024. URL https://arxiv.org/abs/2402. 03300.

Yuhao Shen, Junyi Shen, Quan Kong, Tianyu Liu, Yao Lu, and Cong Wang. Specbranch: Speculative decoding via hybrid drafting and rollback-aware branch parallelism. In The Fourteenth International Conference on Learning Representations, 2026.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. ArXiv preprint, abs/2409.19256, 2024. URL https://arxiv.org/abs/2409.19256.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. ArXiv preprint, abs/1909.08053, 2019. URL https://arxiv.org/abs/1909.08053.

THUDM. Slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025.

Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, Yancheng He, Ju Huang, Jiaheng Liu, Zhendong Li, Xiaoyang Li, Zichen Liu, Haizhou Zhao, et al. Reinforcement learning optimization for large-scale learning: An efficient and user-friendly scaling library. ArXiv preprint, abs/2506.06122, 2025. URL https://arxiv.org/abs/2506.06122.

Tianyu Wu, Yu Yao, Zhenting Qi, Han Zheng, Zhuohan Wang, Haoran Ma, Lawrence Liao, Himabindu Lakkaraju, Ju Li, and Yilun Du. D-pace: Dynamic position-aware cross-entropy for parallel speculative drafting. ArXiv preprint, abs/2605.18810, 2026. URL https://arxiv.org/abs/2605.18810.

Bangjun Xiao, Tianyang Lu, Weiji Zhuang, et al. MiMo-V2-Flash technical report. ArXiv preprint, abs/2601.02780, 2026. URL https://arxiv.org/abs/2601.02780.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. ArXiv preprint, abs/2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, 2025. URL https://fengyao.notion.site/ off-policy-rl.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025. Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu,

Rui Men, An Yang, et al. Group sequence policy optimization. ArXiv preprint, abs/2507.18071, 2025. URL https://arxiv.org/abs/2507.18071.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2023, 2023.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs. In NeurIPS, 2024, 2024.

- A Derivation of TV Loss Gradient We provide the full derivation of the TV loss gradient (Eq. (11)).

zj

Let the draft head output logits z ∈ R|V| with qj = softmax(z)j = e

∑k ezk . The target model probability p is treated as a constant (detached). The TV loss is:

LTV = 1−∑

min(pv, qv).

v

The gradient with respect to zj is:

∂LTV ∂zj

∂

### ∂zj ∑

min(pv, qv).

= −

v

Since p is constant, the subgradient of min(pv, qv) with respect to qv is:

∂ ∂qv

min(pv, qv) = [q v ≤ pv].

Using the chain rule with the softmax Jacobian ∂∂qzv

= qv(δvj − qj): ∂

j

### ∂zj ∑

min(pv,qv) = ∑

[q v ≤ pv] · qv(δvj − qj)

v

v

### −qj ∑

[q v ≤ pv] · qv

###### = [q j ≤ pj] · qj

v

v=j term

###### ≜ S

= qj [q j ≤ pj] − S . Therefore:

∂LTV ∂zj

###### = −qj [q j ≤ pj] − S ,

where S = ∑v [q v ≤ pv] · qv ∈ [0,1].

Boundedness. Since qj ∈ [0,1] and |[q j ≤ pj] − S| ≤ 1:

∂LTV ∂zj

= qj · |[q j ≤ pj] − S| ≤ qj ≤ 1.

#### B Comparison with Forward KL Divergence Gradient

For comparison, the gradient of the forward KL divergence DKL(p∥q) = ∑v pv log qpv

with respect to zj is: ∂DKL(p∥q) ∂zj

v

= qj − pj.

Key differences from the TV loss gradient:

- 1. The forward KL gradient applies a nonzero force to every token where qj ̸= pj, including tokens with negligible probability. The TV gradient is proportional to qj, so it automatically ignores low-probability tokens.
- 2. The forward KL gradient does not distinguish between tokens that would be accepted vs. rejected under rejection sampling. The TV gradient explicitly incorporates this distinction via the indicator [q j ≤ pj].
- 3. The forward KL gradient can be large when qj ≫ pj (overconfident draft). The TV gradient is bounded by qj.

#### C Analysis of the Reverse KL Divergence

The preceding analysis focuses on the forward KL divergence DKL(p∥q), which is equivalent to CE loss up to a constant. A natural question is whether the reverse KL divergence DKL(q∥p) = ∑v qv log qpv

v

would be a better training objective for rejection sampling.

Gradient derivation. The gradient of the reverse KL divergence with respect to the draft logits zj is:

∂DKL(q∥p) ∂zj

### = ∑

log(qv/pv) + 1 · qv(δvj − qj)

v

= qj log(qj/pj) +1 −qj ∑

qv log(qv/pv) + 1

v

= qj log(qj/pj) − DKL(q∥p) . (17) Comparison of gradient structures. Table 1 summarizes the three gradient structures.

The reverse KL gradient shares the desirable qj-proportionality with the TV gradient, meaning lowprobability tokens automatically receive negligible optimization pressure. This suggests that the reverse KL should produce a mismatch that scales more proportionally with qj than the uniform mismatch of forward KL, and consequently exhibit weaker entropy–acceptance coupling than the forward KL.

Why reverse KL is still suboptimal. Despite the improved gradient structure, the reverse KL remains suboptimal for maximizing the rejection sampling acceptance rate for three reasons:

- 1. Zero-forcing behavior. The reverse KL does not penalize q(v) → 0 even when p(v) > 0, since limq→0 q log(q/p) = 0. This “mode-seeking” property allows the draft to drop modes of p, directly forfeiting the overlap min(p(v), q(v)) at those tokens and reducing the acceptance rate. In contrast,

the forward KL is “zero-avoiding” (DKL(p∥q) → ∞ when q(v) → 0 with p(v) > 0), enforcing full support coverage. The TV loss is neither zero-forcing nor zero-avoiding: it selectively allocates capacity to tokens where the marginal overlap improvement is largest.

- 2. Asymmetric over-/under-estimation penalty. The acceptance ratio of rejection sampling depends

on ∑v min(p(v), q(v)), which penalizes over-estimation (q > p) and under-estimation (q < p) symmetrically—both reduce the overlap by |q(v) − p(v)|. The reverse KL imposes an asymmetric

penalty: over-estimation (qj > pj, so log(qj/pj) > 0) incurs a much stronger gradient than underestimation. This drives the draft toward q(v) ≤ p(v) across most tokens, which ensures individualtoken acceptance probability min(1, p/q) = 1 but reduces the sampling probability of those tokens, yielding suboptimal total overlap.

- 3. Indirect optimization target. Like the forward KL, the reverse KL does not directly optimize dTV(p, q). The log-ratio log(qj/pj) in the reverse KL gradient provides a soft, nonlinear signal,

whereas the TV gradient’s indicator [q j ≤ pj] provides a hard, direct signal aligned with the rejection sampling decision boundary.

Summary. In terms of suitability for optimizing rejection sampling acceptance rates:

TV loss > Reverse KL > Forward KL (CE).

The reverse KL improves upon the forward KL through better capacity allocation (gradient ∝ qj), but remains suboptimal due to its zero-forcing behavior and asymmetric penalty structure. The TV loss directly optimizes the quantity of interest and avoids both failure modes.

#### D Entropy-Acceptance Relationship under Different Training Objectives

We provide a detailed analysis of how the target model’s entropy H(p) constrains MTP acceptance rates under different acceptance methods and training objectives.

###### D.1 Setup and Notation

Consider a fixed position t in the generation process. Let p ∈ ∆|V| denote the target model’s distribution and q ∈ ∆|V| the draft model’s distribution. The draft model is parameterized by qθ with logits z ∈ R|V| and qj = softmax(z)j. Due to finite model capacity, the draft cannot perfectly match p in general, and the per-token mismatch structure depends critically on the training objective.

We define the effective support of p at threshold τ as Sτ(p) = {v ∈ V : p(v) > τ}, and recall that the effective support size is related to entropy via the perplexity: |Seff(p)| ≈ exp(H(p)).

For the analysis below, we consider two mismatch structures depending on the training objective:

- • Uniform mismatch (CE/KL training): q∗(v) = p(v) + ηv with |ηv| ≲ σ and ∑v ηv = 0, where σ is approximately uniform across tokens (see §D.3 for justification).
- • Probability-proportional mismatch (TV training): |q∗(v) − p(v)| ≲ δ · p(v), where the absolute error scales with the token probability (derived under the capacity-allocation assumption in §D.4).

###### D.2 Target-Only Sampling

Under target-only sampling, the draft token is selected greedily as yˆ = argmaxy q(y) and accepted with probability p(yˆ), giving acceptance rate:

αTO = p argmax

q(y) . (18)

y

Perfect draft case. For a well-trained draft model where argmaxy q(y) = argmaxy p(y) (i.e., the draft correctly identifies the target’s top-1 token), the acceptance rate reduces to:

αTO = max

p(y). (19)

y

Relationship to Shannon entropy. The quantity maxy p(y) is a monotonically decreasing function of H(p): as entropy increases, the distribution spreads and the maximum probability decreases. A standard bound gives maxy p(y) ≥ exp(−H(p)), so the acceptance rate is lower-bounded by exp(−H(p)).

Linearization. Since αTO = maxy p(y) is a smooth, monotonically decreasing function of H(p), we can write αTO = f(H(p)) for some decreasing function f. Performing a first-order Taylor expansion around

the mean operating entropy H¯ = 21(Hmin + Hmax): αTO = f(H) ≈ f(H¯ ) + f′(H¯ ) · (H(p) − H¯ )

= f(H¯ ) − f′(H¯ )H¯

###### + f′(H¯ )

·H(p). (20)

−bTO

aTO

Since f is decreasing, f′(H¯ ) < 0, so bTO = −f′(H¯ ) > 0, yielding: αTO ≈ aTO − bTO · H(p). (21)

The lower bound f(H) ≥ exp(−H) provides an order-of-magnitude estimate for the slope: bTO ∼ exp(−H¯ ). Empirically, this linear approximation is remarkably robust across model scales, tasks, and training stages (Fig. 1a).

Imperfect draft correction. With an imperfect draft under uniform per-token mismatch, a ranking error argmax q ̸= argmax p occurs when the gap between the top two target probabilities satisfies p(v1∗) − p(v2∗) ≲ 2σ. High-entropy distributions have smaller gaps among top tokens, making ranking errors more frequent. When a ranking error occurs, the acceptance rate drops from p(v1∗) to p(vˆ) < p(v1∗), introducing an additional entropy-dependent deficit. Both effects reinforce the negative slope, so the linear approximation still holds with a potentially steeper slope:

αTO ≈ aTO − bTO · H(p), (22) where the slope bTO is empirically comparable to bRS (see §6), though the two arise from different mechanisms: bTO is driven by the concentration of maxy p(y) and ranking instability, while bRS is driven by the accumulation of per-token TV residuals.

###### D.3 Rejection Sampling with CE/KL Training

The rejection sampling acceptance rate is αRS = 1 − dTV(p, q) (Eq. (2)). We analyze how CE/KL training produces entropy-dependent acceptance rates through its uniform per-token mismatch structure.

Gradient structure. The gradient of DKL(p∥q) with respect to logits is ∂D∂zKL

= qj − pj (see §B). The gradient magnitude |qj − pj| is determined by the absolute difference between pj and qj, not by the magnitude of pj itself. Under gradient-based optimization, each token receives optimization pressure proportional to |qj − pj|, regardless of whether pj = 10−1 or pj = 10−5. This uniform pressure produces approximately uniform per-token mismatch: qCE∗ (v) = p(v) + ηv with |ηv| ≲ σ.

j

TV distance derivation. Under uniform per-token mismatch:

- 1

- 2 ∑

- 1

- 2 ∑

|p(v) − q∗(v)| =

dTV(p, qCE∗ ) =

|ηv|. (23)

v

v∈V

The sum decomposes over the effective support Sτ(p) and its complement: dTV =

- 1

- 2 ∑

- 1

- 2 ∑

|ηv|. (24)

|ηv| +

v∈Sτ

v∈S/ τ

For the complement term, since p(v) ≈ 0 outside the effective support and q∗(v) ≥ 0, we have |ηv| = |q∗(v) − p(v)| ≤ q∗(v), so:

- 1

- 2 ∑

- 1

- 2

- 1

- 2 ∑

### 1− ∑

q∗(v) ≤

q∗(v) , (25)

|ηv| ≤

v∈Sτ

v∈S/ τ

v∈S/ τ

which is a small constant independent of H(p) (since most probability mass concentrates in the effective support for both p and q∗). The entropy-dependent contribution therefore comes from the effective support, where mismatch is fully realized at the σ level. With |ηv| ≲ σ for v ∈ Sτ and |Sτ(p)| ≈ exp(H(p)):

σ 2 · exp(H(p)). (26)

dTV(p, qCE∗ ) ≈

Therefore:

αRSCE = 1 − dTV ≈ 1 −

σ 2

exp(H(p)). (27)

Linear approximation. In the regime where H(p) varies over a moderate range [Hmin, Hmax] (e.g., [0.1,0.5] during RL training), the exponential can be linearized via a first-order Taylor expansion around

###### H¯ = 12(Hmin + Hmax):

exp(H(p)) ≈ exp(H¯ ) · 1 + (H(p) − H¯ ) . (28) Substituting:

αRSCE ≈ aRS − bRS · H(p), (29)

where aRS = 1 − σ2 exp(H¯ )(1 − H¯ ) and bRS = σ2 exp(H¯ ) are positive constants. This explains the empirically observed linear negative correlation between entropy and acceptance rate under CE/KL

training.

Intuition. CE/KL training distributes optimization resources uniformly across all tokens. When H(p) is low, p concentrates on a few tokens, and the draft only needs to match these accurately — the additive errors on the remaining tokens contribute negligibly to dTV. When H(p) is high, p spreads across exp(H(p)) tokens, and the uniform additive errors accumulate into a large TV distance.

##### Why CE/KL training is suboptimal for rejection sampling. Pinsker’s inequality states dTV(p, q) ≤

- 1

- 2DKL(p∥q), relating the two divergences. However, the suboptimality of CE/KL training for rejection

sampling does not stem from the looseness of this bound per se, but from how the KL gradient allocates model capacity across the vocabulary.

Under uniform per-token mismatch, a second-order expansion of DKL gives:

ηv2 p(v)

- 1

- 2 ∑

DKL(p∥qCE∗ ) ≈

, (30)

v

- 1

- 2 ∑

dTV(p, qCE∗ ) =

|ηv|. (31)

v

By the Cauchy–Schwarz inequality, (∑v |ηv|)2 ≤ ∑v ηv2/p(v) (∑v p(v)), which recovers Pinsker’s bound (2 dTV)2 ≤ 2 DKL. Equality holds when |ηv| ∝ p(v)—i.e., the bound is tightest when p is uniform.

The fundamental issue is instead one of capacity allocation. The KL gradient ∂DKL/∂zj = qj − pj applies optimization pressure proportional to the absolute difference |qj − pj|, distributing finite model capacity roughly uniformly across all tokens, including those with negligible target probability. Under this uniform allocation, each token contributes a uniform mismatch |ηv| ≲ σ, and the resulting TV distance scales with the number of tokens in the effective support:

σ 2 · |Seff(p)| ∝ exp(H(p)) · σ. (32)

dTV ≈

High-entropy distributions spread mass across more tokens (|Seff| ≈ exp(H)), accumulating more pertoken residuals into a larger TV distance, even though the KL divergence is also being minimized. This is why CE/KL-trained drafts exhibit a strong negative entropy–acceptance correlation: the KL objective does not distinguish between tokens that matter for the rejection sampling acceptance decision and those that do not.

###### D.4 Rejection Sampling with TV Training

Gradient structure. The gradient of the TV loss with respect to logits is ∂L∂zTV

j

(Eq. (11)). Key observation: The gradient is proportional to qj. This means:

###### = −qj[[q j ≤ pj] − S]

- • High-probability tokens (qj large) receive a strong gradient signal and are optimized accurately.
- • Low-probability tokens (qj ≈ 0) receive near-zero gradient, so the optimizer does not waste capacity on them.

TV gradient as a self-correcting mechanism. Define the probability ratio rj = qj/pj. The TV gradient (Eq. (11)) acts as a self-correcting feedback that drives rj → 1:

- • When rj < 1 (i.e., qj < pj): the indicator [q j ≤ pj] = 1, so ∂LTV

∂zj

= −qj(1 − S) < 0, (33)

and gradient descent increases zj, pushing qj upward and rj toward 1.

- • When rj > 1 (i.e., qj > pj): the indicator [q j ≤ pj] = 0, so

∂LTV ∂zj

= qj · S > 0, (34)

and gradient descent decreases zj, pushing qj downward and rj toward 1.

In both cases, TV training drives rj → 1, i.e., log(qj/pj) → 0. Moreover, since qj ≪ 1 for typical vocabulary sizes, the softmax locally satisfies ∂qj/∂zj ≈ qj, so a single gradient step produces

η qj(1 − S) > 0 if rj < 1, −η qj S < 0 if rj > 1,

∆(logrj) ≈ ∆zj =

(35)

where η is the learning rate. The correction magnitude is proportional to qj: tokens with larger probability receive a stronger corrective signal, ensuring that | logrj| on the effective support converges to a bounded value ϵ. Tail tokens (qj ≈ 0) receive negligible correction but also contribute negligible TV distance.

This self-correcting dynamics contrasts with CE/KL training, whose gradient ∂DKL/∂zj = qj − pj drives absolute differences |qj − pj| toward zero uniformly, rather than ratios qj/pj toward one. Under finite capacity, the CE/KL equilibrium maintains |qj − pj| ≲ σ uniformly, which corresponds to |qj/pj − 1| ≲ σ/pj—an unbounded ratio for small-pj tokens in the effective support.

Assumption: bounded logit-ratio error on the effective support. The self-correcting property above motivates the following assumption. Let qTV∗ be the solution reached by TV training under finite draft capacity. Since the correction magnitude in Eq. (35) is proportional to qj ≈ pj for tokens in the effective support, these tokens receive sufficient gradient signal to drive logrj into a bounded interval.

The assumption is stated in log-ratio space (| log(q/p)| ≤ ϵ) rather than absolute space (|q − p| ≤ σ) because gradient descent operates on logits zj, and the softmax satisfies log qj = zj − log Z, so each logit update ∆zj directly translates to ∆(log qj) ≈ ∆zj. Since p is fixed, ∆(logrj) = ∆(log qj) ≈ ∆zj: the optimizer’s native space is log-ratio, and the equilibrium error is therefore naturally bounded in log-ratio.

We assume: there exists a constant ϵ such that, for all j ∈ Seff(p),

qTV∗ (j) pj ≤ ϵ. (36)

log

Tail tokens may have larger relative uncertainty but carry negligible probability mass and contribute negligible TV distance.

Deriving the mismatch bound. The bounded logit-ratio assumption implies

e−ϵpj ≤ qTV∗ (j) ≤ eϵpj. (37) Therefore, for every token in the effective support,

Letting δ = eϵ − 1, we obtain

qTV∗ (j)

|qTV∗ (j) − pj| = pj

pj − 1 (38) ≤ pj max{eϵ − 1, 1 − e−ϵ} (39) = (eϵ − 1)pj. (40)

||qTV∗ (j) − pj| ≲ δ pj.|
|---|

(41)

That is, under the bounded-logit-ratio assumption induced by the TV gradient’s capacity allocation, TV training yields probability-proportional mismatch (|q − p| ≲ δ · p) rather than the uniform mismatch (|q − p| ≲ σ) of CE/KL training. In practice, optimizer dynamics (e.g., Adam’s second-moment normalization) may partially attenuate the raw qj-proportionality, so the proportional mismatch should be viewed as a modeling approximation rather than an unconditional theorem.

TV distance derivation. Under probability-proportional mismatch with constant δ:

- 1

- 2 ∑

δ

### 2 ∑

dTV(p, qTV∗ ) =

|p(v) − q∗(v)| ≤

v

v

p(v) =

δ 2

. (42)

This bound is independent of H(p), yielding:

δ 2

αRSTV ≥ 1 −

, (43) which proves Proposition 4.

Practical considerations. The above analysis assumes δ is a constant, but in practice, the draft head has finite capacity. When H(p) increases, the effective support |Seff(p)| ≈ exp(H(p)) grows, and maintaining uniform relative accuracy across more tokens may require more model capacity. If the draft head’s capacity is insufficient, δ may exhibit weak entropy dependence δ = δ(H), reintroducing a residual (but substantially attenuated) entropy–acceptance correlation. Empirically, the entropy–acceptance slope under TV training is reduced by over 95% compared to CE/KL training (e.g., −0.06 vs. −1.68), confirming that the probability-proportional mismatch largely holds but is not perfect.

Intuition. TV training allocates optimization resources proportionally to each token’s probability. When H(p) is high and the distribution spreads across many tokens, each token receives proportionally less optimization effort, but also carries proportionally less weight in the TV distance. These two effects largely cancel, making the entropy–acceptance relationship substantially weaker than under CE/KL training.

#### E Rejection Sampling Decision Boundary Derivation

We derive the condition under which rejection sampling achieves a higher acceptance rate than target-only sampling.

Acceptance rates. Under target-only sampling, the acceptance rate is αTO = p(yˆ), where yˆ = argmaxy q(y) is the draft’s top-1 token. Under rejection sampling, the acceptance rate is:

### αRS = ∑

min p(v), q(v) . (44)

v∈V

Decomposing αRS. Using the identity min(a, b) = 12(a + b − |a − b|) and the normalization ∑v p(v) = ∑v q(v) = 1:

###### p(v) + q(v) − |p(v) − q(v)| 2

### αRS = ∑

(45)

v

- 1

- 2∑

- 1

- 2∑

- 1

- 2∑

|p(v) − q(v)| (46)

p(v) +

q(v) −

=

v

v

v

= 1 − dTV(p, q). (47) Decision boundary. RS outperforms target-only when αRS > αTO:

1 − dTV(p, q) > p(yˆ) ⇐⇒ dTV(p, q) < 1 − p(yˆ). (48) This reduces the comparison between the two acceptance methods to a simple inequality: RS is preferred whenever the draft–target TVD is smaller than the target probability mass outside the draft’s greedy prediction. Since 1 − p(yˆ) ≥ 1 − maxy p(y) > 0 for any non-degenerate distribution, there always exists a sufficiently well-aligned draft for which RS is beneficial.

#### F Fused TV Loss Kernel

We provide the pseudocode for our fused TV loss implementation. The forward pass (Algorithm 1) computes the per-token TV loss and the auxiliary quantity S needed by the backward pass in a single kernel launch. The backward pass (Algorithm 2) computes gradients with respect to the draft logits. Both kernels iterate over the vocabulary in tiles of size BLOCK V to bound register and shared-memory usage, enabling full-vocabulary TV loss computation without materializing the softmax output.

- Algorithm 1 TV Loss Forward Kernel (per token position)

Require: Draft logits z ∈ R|V|, target log-probs log p ∈ R|V| Ensure: TV loss ℓ, auxiliary scalar S

- 1: // Pass 1: numerically stable softmax denominator
- 2: m ← maxv zv ▷ global logit max
- 3: D ← ∑v exp(zv − m) ▷ exp-sum
- 4: // Pass 2: tiled overlap and S accumulation
- 5: overlap ← 0; S ← 0
- 6: for vstart = 0 to |V| step BLOCK V do

- 7: v ← [vstart, . . . , vstart + BLOCK V−1]

- 8: q ← exp(z[v] − m) / D ▷ draft prob
- 9: p ← exp(log p[v]) ▷ target prob
- 10: overlap += ∑ min(q, p)
- 11: S += ∑ q · [q ≤ p]
- 12: end for
- 13: ℓ ← clamp(1 − overlap, 0, τmax) ▷ τmax: optional clamp
- 14: return ℓ, S

- Algorithm 2 TV Loss Backward Kernel (per token position)

Require: Draft logits z, target log-probs log p, cached (m, D, S, gout) Ensure: Gradient ∇zℓ ∈ R|V|

- 1: for vstart = 0 to |V| step BLOCK V do

- 2: v ← [vstart, . . . , vstart + BLOCK V−1]

- 3: q ← exp(z[v] − m) / D
- 4: p ← exp(log p[v])
- 5: ∇z[v] ← q · (S − 1 + [q > p]) · g out
- 6: end for
- 7: return ∇z

Implementation notes. (1) The forward kernel fuses the softmax normalization with the TV overlap computation, avoiding a separate O(|V|) softmax pass. (2) For tensor-parallel training, m and D are computed via all reduce across TP ranks before the overlap pass; the local overlaps and S values are similarly reduced after computation. (3) The optional top-K path selects the K largest draft logits and computes TV/gradients only at those positions, reducing memory from O(|V|) to O(K) with negligible accuracy loss (since the gradient ∝ qj ≈ 0 for tail tokens).

#### G Rejection Sampling Inference Implementation

Implementing rejection sampling for MTP-based speculative decoding in production inference engines requires modifying both the draft and verification stages. Unlike target-only sampling, which selects draft tokens via argmax and accepts based solely on the target probability, rejection sampling requires (1) sampling draft tokens from the draft distribution q (rather than taking the argmax), (2) caching the draft probabilities for use during verification, and (3) computing the acceptance ratio min(1, p(yˆ)/q(yˆ)) during verification. We describe two different implementation strategies as follows.

###### G.1 Multinomial Draft Sampling (SGLang)

The first approach, implemented in SGLang5, directly samples draft tokens from the draft distribution using multinomial sampling.

5https://github.com/sgl-project/sglang/pull/26312

Draft stage. Instead of selecting draft tokens via yˆ = argmaxy q(y), we apply temperature scaling to the draft logits and sample yˆ ∼ q(·) via multinomial sampling. The full draft probability vector q ∈ R|V| is cached alongside each draft token for use during verification.

Verification stage. Given a chain of γ draft tokens yˆ1, . . . , yˆγ with cached draft probabilities q1, . . . , qγ, and the target probabilities p1, . . . , pγ obtained from the single-pass target model verification, we implement rejection sampling via a fused Triton kernel. The kernel processes each request independently (one Triton program per request) and performs two phases:

- 1. Sequential acceptance: For each draft step i = 1, . . . , γ, draw ui ∼ Uniform(0,1) and accept yˆi if ui · qi(yˆi) < pi(yˆi), i.e., with probability min 1, pi(yˆi)/qi(yˆi) . Stop at the first rejection.
- 2. Residual resampling: If draft token yˆj is rejected at step j, or if all γ drafts are accepted (bonus token case), sample the next token from the residual distribution. For rejection at step j, the residual distribution is presid(v) ∝ max(0, pj(v) − qj(v)); for the bonus token (all accepted), the residual is simply pγ(v). The kernel computes this via a two-pass CDF inversion over the vocabulary: Pass 1 computes the normalization constant Z = ∑v max(0, pj(v) − qj(v)), and Pass 2 finds the token v∗ such that the cumulative sum first exceeds u · Z for a uniform random u.

- Algorithm 3 Chain Rejection Sampling Verification (Multinomial / SGLang)

Require: Draft tokens yˆ1, . . . , yˆγ; draft probs q1, . . . , qγ ∈ R|V|; target probs p1, . . . , pγ ∈ R|V| Ensure: Accepted token count n; output token y∗ at position n + 1

- 1: n ← γ ▷ assume all accepted
- 2: for i = 1 to γ do
- 3: ui ∼ Uniform(0,1)
- 4: if ui · qi(yˆi) ≥ pi(yˆi) then ▷ reject
- 5: n ← i − 1; break
- 6: end if
- 7: end for
- 8: // Residual resampling via two-pass CDF inversion
- 9: if n < γ then ▷ rejected at step n + 1
- 10: r(v) ← max 0, pn+1(v) − qn+1(v) for all v
- 11: else ▷ bonus token
- 12: r(v) ← pγ(v) for all v
- 13: end if
- 14: Z ← ∑v r(v) ▷ Pass 1: normalization
- 15: u ∼ Uniform(0,1)
- 16: y∗ ← min v : ∑v′≤v r(v′) ≥ u · Z ▷ Pass 2: CDF inversion
- 17: return n, y∗

Memory overhead. The primary overhead is caching the draft probability vectors: O(γ × |V|) per request, where γ is the number of MTP steps.

###### G.2 Gumbel-Max Trick (vLLM)

The second approach, implemented in vLLM6, avoids explicit CDF inversion during residual resampling by leveraging the Gumbel-Max trick.

Draft stage. Draft tokens are sampled using the Gumbel-Max trick: for each vocabulary token v, compute v∗ = argmaxv[log q(v)/τ + Gv], where Gv ∼ Gumbel(0,1) is i.i.d. Gumbel noise and τ is the sampling temperature. This is equivalent to sampling from q after temperature scaling. The temperaturescaled draft logits (before adding Gumbel noise) are cached for verification.

Verification stage. The verification is split into two kernels:

- 1. Acceptance kernel: A sequential Triton kernel iterates over draft steps, computing p(yˆi) and q(yˆi) from the cached target and draft probabilities, and accepting if ui · q(yˆi) < p(yˆi) for a pseudo-

random ui generated via tl.rand seeded by the request’s random seed and position. The kernel records the index of the first rejected step.

6https://github.com/vllm-project/vllm/pull/35461

- 2. Residual logits kernel: A parallel Triton kernel computes the residual distribution in logit space. For

rejection at step j: zresid(v) = logmax(0, pj(v) − qj(v)); for the bonus token: zresid(v) = ztarget,γ(v) (the raw target logits). The resampled token is then drawn from this residual distribution using the same Gumbel-Max sampling as the draft stage.

- Algorithm 4 Chain Rejection Sampling Verification (Gumbel-Max / vLLM)

Require: Draft tokens yˆ1, . . . , yˆγ; draft logits z1q, . . . , zqγ ∈ R|V|; target probs p1, . . . , pγ ∈ R|V|; target

logits zγp

Ensure: Accepted token count n; output token y∗ at position n + 1

- 1: // Kernel 1: sequential acceptance
- 2: n ← γ
- 3: for i = 1 to γ do
- 4: qi(yˆi) ← softmax(ziq)yˆi
- 5: ui ← tl.rand(seed,i)
- 6: if ui · qi(yˆi) ≥ pi(yˆi) then
- 7: n ← i − 1; break
- 8: end if
- 9: end for
- 10: // Kernel 2: residual logits
- 11: if n < γ then
- 12: zresid(v) ← logmax 0, pn+1(v) − qn+1(v) for all v
- 13: else
- 14: zresid(v) ← zγp(v) for all v
- 15: end if
- 16: // Gumbel-Max resampling
- 17: Gv ∼ Gumbel(0,1) for all v
- 18: y∗ ← argmaxv zresid(v) + Gv
- 19: return n, y∗

