## LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding

Alexander Samarin1 Sergei Krutikov1 Anton Shevtsov1 Sergei Skvortsov1 Filipp Fisin1 Alexander Golubev1

# arXiv:2602.23881v2[cs.LG]1Jun2026

### Abstract

Speculative decoding accelerates autoregressive large language model (LLM) inference by using a lightweight draft model to propose candidate tokens that are then verified in parallel by the target model. The speedup is significantly determined by the acceptance rate, yet standard training minimizes Kullback-Leibler (KL) divergence as a proxy objective. While KL divergence and acceptance rate share the same global optimum, small draft models, having limited capacity, typically converge to suboptimal solutions where minimizing KL does not guarantee maximizing acceptance rate. To address this issue, we propose LK losses, special training objectives that directly target acceptance rate. Comprehensive experiments across four draft architectures and six target models, ranging from 8B to 685B parameters, demonstrate consistent improvements in acceptance metrics across all configurations compared to the standard KL-based training. We evaluate our approach on general, coding and math domains and report gains of up to 8-10% in average acceptance length. LK losses are easy to implement, introduce no computational overhead and can be directly integrated into any existing speculator training framework, making them a compelling alternative to the existing draft training objectives.

### 1. Introduction

Large language model (LLM) inference is fundamentally constrained by memory bandwidth rather than computational throughput. The autoregressive nature of token generation requires sequential memory accesses that underutilize modern accelerators, creating a critical bottleneck for deployment at scale. Speculative decoding (Leviathan et al.,

1Nebius, Amsterdam, Netherlands. Correspondence to: Alexander Golubev <alex golubev@nebius.com>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

[Figure 1]

Figure 1. Acceptance length τ vs maximum length K for EAGLE3 draft models, trained using different objectives with Qwen3235B-A22B-Instruct as a target model. The values were obtained on the MT-Bench dataset with chain sampling at temperature = 1.

2023; Chen et al., 2023) addresses this challenge through a draft-then-verify paradigm: a lightweight draft model proposes multiple candidate tokens, which the target model then verifies in a single forward pass. This approach preserves the output distribution of the target model while achieving substantial speedups. The efficiency of this process is largely determined by the acceptance rate – an expected probability of a drafted token being accepted by the target model.

A variety of draft model architectures implementing that idea have emerged, including parallel prediction heads (Cai et al., 2024; Sandler et al., 2025), autoregressive draft heads with feature fusion (Li et al., 2024; 2025b), and multitoken prediction modules that can be repurposed for speculation after training (DeepSeek-AI et al., 2024). These approaches commonly train draft models by minimizing Kullback-Leibler (KL) divergence between target and draft distributions, treating distributional alignment as a proxy for acceptance rate optimization. At the global optimum draft matches target perfectly, so this proxy is exact: KL divergence reaches zero and acceptance rate reaches one. However, draft models operate under severe capacity constraints, typically having 1-5% of the target model parameters, and

inevitably converge to suboptimal solutions. At these suboptimal points, minimizing KL divergence provides no guarantee of maximizing acceptance rate. Leviathan et al. (2023) noted that greater improvements might be obtained via custom training procedures, including direct optimization of the draft model for the acceptance rate, yet this direction has remained largely unexplored.

In this work, we propose two variants of LK losses 1, training objectives that directly target acceptance rate. One of them is motivated by the maximum likelihood methods, directly optimizing the negative log-acceptance rate. The other variant is a hybrid approach, which gradually shifts the focus from KL towards direct acceptance optimization as training progresses, analogous to trust-region methods that balance a stable surrogate objective against the true optimization target. Both losses help improving average acceptance length, especially for longer draft sequences (see Figure 1).

In summary, our contributions are as follows:

- • We propose two loss variants for direct acceptance rate optimization in speculative decoding.
- • We demonstrate consistent improvements in acceptance metrics across a number of target models and draft architectures, empirically confirming that LK losses are both model- and architecture-agnostic.
- • We release our training datasets and draft model weights to facilitate reproducibility and further research 2.

MEDUSA (Cai et al., 2024) attaches parallel decoding heads to the target model’s final layer to predict draft tokens independently from each other. This approach is computationally efficient but implicitly assumes conditional independence between the proposed tokens, which can degrade performance for distant draft positions.

Other methods introduce autoregressive draft heads that also operate in the target model’s hidden state space. Wertheimer et al. (2024) propose a multi-stage MLP speculator that extends MEDUSA heads with the principles from recurrent networks. A more advanced EAGLE (Li et al., 2024; 2025b) family employs a shallow transformer model with a causal mask to better capture long-term dependencies between draft tokens and their context. EAGLE-3 also enriches the input feature space of the speculator heads by fusing hidden states from various intermediate layers of the target model.

A similar autoregressive design was recently adopted by DeepSeek-V3 with its Multi-Token Prediction (MTP) module (DeepSeek-AI et al., 2024). It predicts multiple draft tokens during training, essentially serving as a native “draft head” that requires no separate post-training.

In addition to architectural advances, some works address the problem of computational overhead in the LM heads of the draft models. FR-Spec (Zhao et al., 2025) observed that large token vocabularies of contemporary LLMs often make the latency of LM heads dominate in the total latency of the speculator heads, even in such powerful approaches as EAGLE. FR-Spec addresses this issue by truncating the draft vocabulary to a small subset of high-frequency tokens learned on training data.

### 2. Related Work

##### 2.1. Draft Model Architectures

Early works on speculative decoding utilized smaller, standalone versions of the target model as draft models (Leviathan et al., 2023; Chen et al., 2023). Although straightforward to implement, such an approach usually suffers from performance bottlenecks within the draft model. Moreover, to achieve high acceptance rate it either needs a small pretrained language model (LM) from the same family or training from scratch on a sufficiently large corpus of data.

To mitigate these overheads, subsequent works integrated the drafting mechanism directly into the target model.

- 1The naming LK positions the loss as an alternative to KL divergence: standard approaches minimize KL as a proxy for acceptance rate, while LK objectives target acceptance directly. The notation also connects to the DLK ≡ TV divergence in the original speculative decoding paper (Leviathan et al., 2023).
- 2Datasets: HuggingFace nebius/infinity-instruct-completions, Weights: HuggingFace nebius/lk-speculators.

##### 2.2. Training Methodologies for Draft Models

Knowledge Distillation. The problem of training speculators has been predominantly framed as the Knowledge Distillation (KD). Within this paradigm, draft models are students that approximate the teacher’s (i.e. target) output distribution as closely as possible. The standard training objective has therefore been KL divergence, or equivalently cross-entropy (CE), between the target and draft distributions.

Some works experimented with combined objectives. MEDUSA suggests mixing the KL loss with the LM objective for the target model to address the discrepancy between its distribution and static training data. EAGLE (Li et al., 2024) extends KL with regression loss on hidden states of the last layer in an attempt to better match training and inference settings.

Targeting Acceptance Rate. Some studies went beyond the standard KD framework in terms of both training data and objectives. DistillSpec (Zhou et al., 2024) explores other types of divergences, including reverse KL and total varia-

tion (TV) distance, as KD objectives for already pretrained LMs being used as external speculators. Although they note that TV distance should theoretically be the right objective as it directly maximizes acceptance rate, they conclude that the choice of the divergence loss is highly dependent on the task and data being used for KD. AdaSPEC (Hu et al., 2025) further addresses the mismatch between standard KD objectives and speculative decoding efficiency through selective distillation strategies for standalone draft models. In contrast, our work focuses on native draft modules tightly integrated into target models and trained from scratch, rather than on separately pretrained external drafters. Within this setting, we study whether the core training objective should optimize token acceptance directly instead of proxy divergence measures.

### 3. Background and Motivation

##### 3.1. Speculative Decoding

In this paper we are working under the standard lossless speculative sampling setting. A draft model q proposes a sequence of tokens x1,...,xK given the context c that are later verified by the target model p in parallel. Each draft token xi in the sequence is accepted with the probability

- p(xi | c,x<i)

- q(xi | c,x<i)

β(xi | c,x<i) = min 1,

,

where x<i = (x1,...xi−1) is a prefix of the i-th draft token. Drafted tokens are verified in parallel and accepted sequentially. The first rejected token terminates the accepted sequence, discarding all subsequent drafts.

The efficiency of speculative decoding is mainly driven by the acceptance rate defined as an expected acceptance probability for the i-th draft token

αi = Ex∼q(·|c,x

<i) β(x | c,x<i)

min(q(x | c,x<i), p(x | c,x<i)), (1)

=

x∈V

where V is the vocabulary of tokens. In the following sections we omit the conditioning and subscripts for brevity wherever the context is clear. From the definition of α above it follows that its global optimum is achieved at q = p.

##### 3.2. Divergence Losses

Statistical divergences are defined as discrepancy measures between probability distributions (Amari & Nagaoka, 2000). As it is noted in Section 2.2, they are widely employed as primary training objectives for draft models. The most commonly used divergences (Zhou et al., 2024) include forward KL divergence,

KL(p∥q) =

i

pi log(pi/qi),

##### reverse KL divergence KL(q∥p) and Total Variation distance

- 1

- 2 i |pi − qi|.

TV(p,q) =

The TV distance is of great interest for us due to its direct relation to acceptance rate via α = 1−TV(p,q) (Leviathan et al., 2023). Thus, maximization of acceptance is strictly equivalent to minimization of TV whereas other divergences serve only as proxy objectives. The gap between proxy and direct optimization becomes stark when model capacity is limited, which is demonstrated in Section 3.3.

Direct influence of TV on acceptance has been noticed and employed in prior works (Zhou et al., 2024; Yin et al., 2024), providing competitive results compared to other training objectives. However, in our setting draft model is initialized randomly rather than from a pretrained language model. We reveal why this distinction is critical in Section 4.1.

##### 3.3. Motivating Example

An important property of the TV distance is its focus on the tokens that constitute the major probability mass under the target distribution (Ji et al., 2023). This is a major advantage given the limited capacity of the draft model and its inability to fully match the target distribution. Figure 2 illustrates this with a simple example – fitting a single Gaussian to a multi-modal Gaussian mixture distribution using different divergences as loss functions. Green areas visualize density overlap, which in fact is equal to the acceptance rate if we were applying speculative sampling algorithm to this toy example (see Appendix C).

Forward KL divergence, being mode-covering, spreads probability mass broadly to avoid infinite penalties wherever the target has support, resulting in suboptimal acceptance rate. Reverse KL divergence exhibits the opposite failure mode, underestimating uncertainty and collapsing to cover only the dominant mode. In contrast, TV distance finds a qualitatively different solution that maximizes the distributional overlap, achieving substantially higher acceptance despite using the same parametric family. This toy example highlights an important observation: when the draft cannot perfectly match the target, the choice of objective determines which compromises the optimization makes.

### 4. Methodology

We propose training objectives that directly target acceptance rate rather than using KL divergence as a proxy. Our approach is grounded in gradient analysis that reveals fundamental differences in how various divergence measures guide optimization.

[Figure 2]

Figure 2. Fitting a single Gaussian to a Gaussian mixture under different objectives. Top: Loss landscapes (log-scale) over parameters µ and σ. Bottom: Resulting distributions and overlap (green). KL divergence produces a mass-covering solution (α = 50.2%), reverse KL exhibits mode-seeking behavior (α = 50.8%), while TV maximizes overlap (α = 60.2%).

##### 4.1. Gradient Analysis of Divergence Losses

Consider training a parametric draft model with logits zq to match a target distribution p through minimization of divergence loss, where q = softmax(zq). The choice of the divergence type fundamentally determines optimization dynamics through its gradient structure. Understanding that is especially important in case when training starts with randomly initialized parameters, which implies large discrepancy between the draft and target distributions. In this section we explore two most relevant divergence losses, forward KL and TV.

The forward KL divergence yields an elegant gradient with respect to the logits (see A.2 for derivations):

KL(p∥q) = q − p. (2)

∇zq

This gradient pushes each logit proportionally to the gap between predicted and target probabilities. At the early stage of the training the gradient magnitude ∥q − p∥ scales as O(1/

√

k) when p is concentrated on k tokens (see A.5), providing strong signal regardless of current alignment. The gradient of TV distance takes more sophisticated form (see A.3)

- 1

- 2

q ⊙ s − Eq[s] , (3)

∇zq

TV(p,q) =

where si = sign(qi − pi) and Eq[s] = i qisi. While this gradient provides directional information about whether

each token is under- or over-predicted, the signal depends only on the sign of the error, not its magnitude. A token with qi slightly below pi receives the same sign signal as one severely under-predicted, which contrasts with KL divergence gradient. The TV loss landscape also contains non-differentiable points along the manifold {zq : qi = pi}, where gradients change discontinuously.

A more significant concern is gradient magnitude. For randomly initialized draft models which spread q over a large vocabulary of size V , the gradient norm satisfies ∥∇zq

√

k/V ) (see A.5). With typical vocabulary sizes exceeding 100k, this yields extremely small gradients at initialization. Together with ignorance of error magnitude and the non-smooth loss landscape, these issues make pure TV optimization impractical for training from random initialization.

TV∥ = O(

##### 4.2. Hybrid Objective with Adaptive Blending

The gradient analysis reveals a fundamental controversy. KL divergence creates smooth, well-conditioned optimization landscapes but optimizes a proxy for the acceptance rate. Conversely, directly minimizing TV distance targets the correct objective but suffers from vanishing gradients and non-smooth optimization surfaces. To benefit from both advantages, we propose a hybrid objective that combines KL divergence with TV distance as follows:

LλLK(p,q) = λ · KL(p∥q) + (1 − λ) · TV(p,q). (4)

With λ = 1 we recover standard KL training whilst setting λ = 0 leads to pure TV optimization.

The key insight is that these components serve complementary roles at different training stages. Early in training, when q is far from p, the KL component provides smooth, properly-scaled gradients that efficiently navigate the loss landscape. As alignment improves, the TV component takes over to directly optimize acceptance rate.

Adaptive schedule. To achieve the aforementioned behavior, we propose the following schedule driven by the ongoing acceptance rate value:

λ = exp −η · sg[α] , η > 0, (5)

where sg[·] denotes stop-gradient operation, preventing backpropagation through λ. We compute λ independently for each draft token position using aggregated values of α across sequence and batch dimensions.

The proposed schedule satisfies the desired properties: λ → 1 when α → 0 (poor alignment), and λ converges to a small value when α → 1 (good alignment), smoothly transitioning from KL-dominated to TV-dominated optimization as training progresses. In Section 6 we empirically confirm through ablation studies that such an adaptive schedule makes the hybrid objective superior to pure TV and KL losses as well as to the fixed mixture of weights.

The hybrid objective with adaptive scheduling can be interpreted through the lens of constrained optimization. When α is small, large λ prioritizes KL minimization, establishing a region where the draft distribution q is sufficiently close to p for TV gradients to behave well. As α increases and λ decays, the objective shifts towards TV minimization while the KL term acts as a soft constraint, maintaining distributional proximity. This resembles the trust-region approach used in policy optimization (Schulman et al., 2015):

TV(p,q) s.t. KL(p∥q) ≤ δ,

min

q

where the adaptive schedule implicitly controls the effective constraint threshold δ based on current alignment quality.

##### 4.3. Likelihood-based Approach

A different interpretation of the acceptance rate comes from its role in the speculative sampling algorithm highlighted in Section 3.1. By definition, β(x) is the conditional probability of acceptance, given that token x was drafted, and q(x) is the probability of drafting token x. Thus, we can interpret

α =

q(x) β(x).

x∈V

as the marginal probability of acceptance. Therefore, it is natural to consider minimization of the negative log

marginal likelihood as a training objective that clearly maximizes α, or more precisely

LαLK(p,q) = −log α

= −log

min(p(x),q(x)).

x∈V

This objective looks appealing due to its simplicity, as it does not need a complex mixture of different losses with adaptive weight scheduling to tackle optimization challenges.

In the degenerate case where p is a point mass, LαLK reduces to the standard negative log-likelihood (see Appendix B for details).

##### Gradient behaviour. We derive in A.4 that

1 α∇zq

∇zqLαLK =

TV(p,q), (6)

which reveals a key insight – LαLK performs TV optimization with adaptive gradient scaling. The 1/α factor provides automatic amplification when acceptance is low (α → 0), addressing the vanishing gradient problem. The gradient magnitude ∥∇zqLαLK∥ matches the one of KL at the early stage of the training (see A.5), while the gradient direction matches that of TV.

We evaluate both LαLK and LλLK empirically in Section 6, finding that both improve over pure TV and KL losses, with the hybrid objective generally achieving stronger results.

##### 4.4. Vocabulary Truncation

EAGLE-3 uses a truncated vocabulary for its LM head as it is proposed in FR-Spec (Zhao et al., 2025), which sets a large subset of draft probabilities to zero. This creates a fundamental problem in KL-based training: KL divergence becomes infinite for tokens outside the draft vocabulary as qi = 0 and pi > 0. To overcome this challenge, we redefine standard target probabilities p = softmax(zp), where zp are the target model logits, as p˜ = softmax(m ⊙ zp), where the mask m sets logits of tokens outside the draft vocabulary to −∞. However, this introduces another layer of approximation as now we optimize KL(˜p∥q) rather than KL(p∥q), making KL a proxy of a proxy.

In contrast, LK losses handle vocabulary truncation naturally. From (1) it follows that tokens outside the draft vocabulary contribute min(pi,0) = 0 to the acceptance rate and therefore have no influence. Thus, no modification of p is needed for LαLK loss and TV component in LλLK which optimize acceptance rate with respect to the original target distribution rather than its approximation.

### 5. Experimental Settings

We evaluate LK losses across a diverse range of target models and draft architectures to assess their generality and

practical impact.

##### 5.1. Target Models

Our experiments span six target models covering three orders of magnitude in parameter count. We include both dense models, namely Llama-3.1-8B-Instruct (Grattafiori et al., 2024) and Llama-3.3-70B-Instruct, and mixtureof-experts (MoE) models, namely gpt-oss-20b, gpt-oss120b, Qwen3-235B-A22B-Instruct (Yang et al., 2025) and DeepSeek-V3 (DeepSeek-AI et al., 2024). This selection allows us to test whether the benefits of LK losses extend to different model scales and architectures.

##### 5.2. Draft Models

We evaluate four speculator architectures which are most commonly present in industrial applications. For Llama3.1-8B we train three architectures to enable direct comparison: EAGLE-3 (Li et al., 2025b), multi-stage MLP speculator (Wertheimer et al., 2024) and MEDUSA (Cai et al., 2024). For larger models (Llama-3.3-70B, gpt-oss, Qwen3), we train only EAGLE-3 with dense transformer block as the best-performing state-of-the-art architecture. For DeepSeek-V3, we fine-tune the native MTP module.

All draft model architectures are trained with K = 6 speculative heads. EAGLE-3 shares weights across positions via recurrence, while the MLP speculator and MEDUSA use fully independent heads at each position. The MTP module retains its original MoE architecture and is initialized from the pretrained DeepSeek-V3 weights.

Rationale for MTP fine-tuning. While our method is primarily designed for training draft heads from scratch, fine-tuning pretrained MTP modules addresses a complementary challenge. Released open-source weights include only the first MTP module, which was originally trained to predict the first token, yet reused autoregressively for later ones. (Liu et al., 2026; Cai et al., 2025). This mismatch causes sharp decline in acceptance rate across later positions. Our adaptive λ scheduler naturally addresses this inconsistency. Early heads with fairly high acceptance receive low λ, focusing on TV-driven adjustments within the trust region to further improve acceptance. Conversely, later heads with degraded acceptance receive higher λ, providing stronger KL guidance precisely where the MTP module was not explicitly trained.

Output vocabulary. EAGLE-3 draft models include a trainable unembedding matrix for next-token prediction. We adopt truncated vocabularies from RedHatAI speculators,3 using only the vocabulary definitions while training all the weights from scratch. For DeepSeek-MTP, we retain the

3https://huggingface.co/RedHatAI

original full vocabulary to preserve compatibility with the pretrained module.

##### 5.3. Training Configuration

We construct the training corpus using 660K prompts from Infinity-Instruct-0625 (Li et al., 2025a) and generating responses with each target model listed above. This ensures the draft model is trained on the same distribution it will encounter during inference. All models are trained with the input sequence length of 8K.

We use batch size of 64 and learning rate 4 × 10−4 with cosine scheduling and 100 warmup steps. Following the original speculative decoding literature, we use AdamW optimizer with (β1, β2) = (0.9,0.95) and gradient clipping at 0.5. All draft models except DeepSeek-MTP are trained from scratch for 10 epochs. For DeepSeek-V3, we initialize from the original MTP weights and fine-tune for 1 epoch. We set temperature T = 1 to match our primary evaluation setting.

We compare the following loss configurations: forward KL divergence KL(p∥q), negative log-acceptance loss LαLK (Section 4.3) and hybrid objective LλLK (Section 4.2) with the adaptive scheduler and η = 3 unless stated otherwise.4 For Llama-3.1-8B with EAGLE-3 we additionally explore total variation distance TV(p,q), LλLK with the fixed weight λ = 0.5 and adaptive LλLK with η = 0.7, 1 and 10.

All the losses are aggregated across draft heads with exponential weight decay γ ∈ (0,1], i.e., the n-th head receives weight γn−1, prioritizing early positions which have the largest impact on average acceptance length. In our experiments we set γ = 0.8, following MEDUSA and EAGLE configurations.

##### 5.4. Evaluation Protocol

Inference Framework. We evaluate draft models using vLLM v0.11.0 (Kwon et al., 2023) with a patch that enables proper rejection sampling at non-zero temperatures.5 In the original version, tokens are sampled greedily from the draft distribution regardless of temperature settings. Our modification implements a theoretically correct rejection sampling procedure from Leviathan et al. (2023). See Appendix D for a detailed analysis of how greedy draft sampling affects acceptance rates.

Evaluation Datasets. We measure acceptance metrics

- 4For MEDUSA we use η = 10 as its acceptance rates improve slower during training compared to recurrent architectures. Larger η accelerates transition towards TV optimization to compensate.
- 5Our patch builds upon https://github.com/ vllm-project/vllm/pull/20459. Equivalent stochastic sampling support was later integrated into upstream vLLM starting from v0.18.0.

on three benchmarks: multi-turn conversational prompts MT-Bench (Zheng et al., 2023), code generation tasks HumanEval (Chen et al., 2021), and grade-school math problems GSM8K (Cobbe et al., 2021). Unlike many prior works that evaluate draft models on small subsets of prompts (e.g. 80 samples in Li et al., 2025b), we evaluate on full datasets to obtain more reliable estimates of the acceptance metrics.

Sampling Configurations. We perform evaluations under two temperature settings: greedy decoding (T = 0) and stochastic sampling (T = 1). Our training objective directly optimizes acceptance probabilities under stochastic sampling, making it the primary evaluation setting. We additionally report greedy decoding results to demonstrate generalization across different sampling regimes.

We evaluate all draft models using chain sampling rather than tree-based drafting (Cai et al., 2024) to isolate the contribution of the training objective from inference-time search optimization. In widely used tree-based speculative decoding methods, such as the EAGLE family, tree construction is typically implemented as a heuristic inference-time procedure on top of a pretrained drafter rather than through a different drafter training objective. Since LK losses directly optimize per-position acceptance rates, improvements are expected to transfer to any verification scheme that relies on these acceptance probabilities.

##### 5.5. Metrics

We report the expected number of tokens generated per speculation round, computed as τ = K × ##accepteddrafted tokenstokens + 1, where K is the maximum draft length. Following standard convention (Leviathan et al., 2023), τ includes the bonus token that is always sampled from the adjusted target distribution after verification, ensuring at least one token is generated per round. This metric is the major driver of the speedup factor of speculative decoding and serves as our primary evaluation criterion.

We evaluate EAGLE-3 and DeepSeek-MTP with K = 7 draft tokens, being consistent with the original EAGLE-3 training and evaluation setup. MEDUSA and MLP speculator are evaluated with K = 6 since weights are not shared between decoding heads in these architectures and generation cannot be extended to longer drafts. We demonstrate in Figure 1 that LK losses improve τ over all values of K.

Additionally we report wall-clock speedup measured as the ratio of tokens per second between speculative and vanilla autoregressive decoding under the same target model and hardware configuration in Appendix F.

Table 1. Average acceptance length τ for LLaMA-3.1-8B-Instruct with EAGLE-3, MEDUSA, and MLP draft models.

MT-Bench HumanEval GSM8K Method Loss τ τ τ

Temperature=0

EAGLE-3 KL 3.75 4.82 4.50 TV 2.81 3.42 3.34 LαLK 3.77 4.82 4.55 LλLK, λ = 0.5 3.78 4.84 4.53 LλLK, η = 0.7 3.79 4.88 4.54 LλLK, η = 1 3.80 4.83 4.54 LλLK, η = 3 3.84 4.89 4.57 LλLK, η = 10 3.67 4.85 4.53

MEDUSA KL 2.05 2.41 2.11 LαLK 2.06 2.42 2.11 LλLK, η = 10 2.07 2.44 2.13

MLP KL 2.45 2.42 2.42 LαLK 2.48 2.46 2.46 LλLK, η = 3 2.48 2.83 2.46

Temperature=1

EAGLE-3 KL 3.39 4.31 3.88 TV 2.67 3.25 3.12 LαLK 3.50 4.48 3.98 LλLK, λ = 0.5 3.35 4.36 3.95 LλLK, η = 0.7 3.53 4.45 3.98 LλLK, η = 1 3.51 4.47 3.96 LλLK, η = 3 3.48 4.52 4.02 LλLK, η = 10 3.34 4.51 4.03

MEDUSA KL 1.72 2.02 1.81 LαLK 1.78 2.09 1.85 LλLK, η = 10 1.85 2.22 1.92

MLP KL 2.13 2.16 2.16 LαLK 2.17 2.19 2.19 LλLK, η = 3 2.19 2.62 2.18

### 6. Evaluation Results

We evaluate LK losses across all target models and draft architectures described in Section 5 with the average acceptance length τ as the primary metric. See Appendix F for a detailed comparison between objectives and against public HuggingFace checkpoints.

##### 6.1. LK Losses across Draft Architectures

Tables 1 presents performance of different draft model architectures trained with various objectives for LLaMA-3.1-8B. The KL baseline demonstrates strong results, but both types of LK losses improve over it across all configurations and sampling temperatures.

Among LK configurations, the hybrid objective with adaptive scheduler achieves the highest acceptance lengths. The likelihood-based objective also outperforms the KL baseline

Table 2. Average acceptance length τ for other target models under different speculative decoding methods. Values in parentheses denote relative improvement (%) of LλLK (η = 3) over KL in average acceptance length. MT: MT-Bench; HE: HumanEval; GSM: GSM8K.

Temperature = 0 Temperature = 1 Model Method/Loss MT (τ) HE (τ) GSM (τ) Mean (∆%) MT (τ) HE (τ) GSM (τ) Mean (∆%) LLaMA 3.1 EAGLE-3 KL 3.75 4.82 4.50 4.36 3.39 4.31 3.88 3.86 8B Instruct EAGLE-3 LλLK 3.84 4.89 4.57 4.43 (+1.6) 3.48 4.52 4.02 4.01 (+3.9) LLaMA 3.3 EAGLE-3 KL 4.01 5.18 5.16 4.78 3.76 4.86 4.89 4.50 70B Instruct EAGLE-3 LλLK 4.00 5.21 5.21 4.81 (+0.5) 3.89 5.08 5.01 4.66 (+3.5) GPT-OSS 20B EAGLE-3 KL 3.31 3.07 4.01 3.46 3.12 2.89 3.51 3.17

EAGLE-3 LλLK 3.35 3.11 4.00 3.49 (+0.9) 3.20 3.01 3.65 3.29 (+3.8) GPT-OSS 120B EAGLE-3 KL 2.81 2.47 3.00 2.76 2.53 2.27 2.57 2.46

EAGLE-3 LλLK 2.86 2.51 3.06 2.81 (+1.8) 2.69 2.44 2.81 2.65 (+7.7) Qwen3-235B-A22B- EAGLE-3 KL 3.33 4.65 4.76 4.25 2.96 4.09 4.27 3.77 Instruct-2507 EAGLE-3 LλLK 3.36 4.74 4.87 4.32 (+1.8) 3.18 4.42 4.65 4.08 (+8.2)

DeepSeek-V3-0324 MTP original 2.90 3.28 3.41 3.20 2.82 3.17 3.27 3.09 (685B) MTP KL 3.96 4.74 5.67 4.79 3.66 4.39 5.23 4.43

MTP LλLK 4.00 4.77 5.72 4.83 (+0.8) 3.88 4.64 5.51 4.68 (+5.6)

though with a smaller margin, particularly under greedy sampling. Yet if the scheduler decay is not optimal (η = 1), the difference between the two LK losses narrows considerably.

Another important observation is that constant weights in the hybrid objective (λ = 0.5) make it inferior to any other LK setting. The advantage of the hybrid loss vs KL loss almost disappears, which confirms that curriculum behavior is necessary for effective training.

Training with pure TV loss performs substantially worse than all other objectives. As derived in Section 4.1, TV gradients have severe optimization difficulties when the draft distribution is far from the target. While TV training converges to a meaningful solution, the resulting acceptance lengths are far from being competitive. The hybrid approach addresses this pathology by using KL gradients to guide optimization to the trust region.

A striking pattern emerges when comparing architectures of varying capacity. MEDUSA and MLP speculators with stochastic sampling show average improvements of 7.8% and 8.3% respectively across domains whereas EAGLE-3 sees 3.8% improvement on average. This aligns with our theoretical analysis: low-capacity draft models benefit more from direct optimization of acceptance rate.

##### 6.2. Scalability across Target Model Sizes

- Table 2 compares performance of our best setting, the hybrid LK loss with η = 3, against the baseline KL objective across target models ranging from 8B to 685B parameter. Our approach provides consistent improvement over KL, regardless of target model architecture and size.

All EAGLE-3 models in this experiment consist of a single dense transformer layer, while target models range from 32 layers (Llama-3.1-8B) to 94 layers (Qwen3-235B). The most significant improvements occur when targeting large MoE models with much smaller dense draft architectures. GPT-OSS 120B shows +7.7% improvement (compared to +3.8% for GPT-OSS 20B) whilst Qwen3-235B achieves the largest gain of +8.2%. We hypothesize that the large relative difference in parameters and the architectural mismatch create capacity gaps that are difficult to overcome via KL divergence alone.

Another remarkable result is achieved for DeepSeek-V3 with stochastic sampling. As it was stated in Section 5.2, released MTP was not trained primarily for predicting multiple tokens ahead. Fine-tuning it with KL loss substantially improves its performance, but LK loss pushes this bound even further with extra 5.6% gain. This result confirms that our approach is superior to KL not only when the draft distribution starts from random, but generally when the discrepancy between the draft and the target is high.

### 7. Conclusion

A standard practice of training draft models for speculative decoding is minimizing KL divergence. It has the same global optimum as acceptance rate, which typically cannot be achieved with capacity-limited draft models. Our approach addresses this gap through LK losses, training objectives that directly target acceptance rate.

We demonstrate through a relevant example that optimizing TV distance provides stronger results than any of the proxy objectives. Our theoretical analysis reveals caveats in gradient-based TV minimization and establishes a solid

ground for formulating objectives that work in practice.

Extensive experiments across six target models, ranging in size from 8B to 685B parameters, and four draft model architectures demonstrate that LK losses consistently improve acceptance metrics. We observe gains up to 10% in average acceptance length across various task domains, with larger improvements for low-capacity architectures. Our approach introduces no computational overhead during training and integrates directly into existing speculator training pipelines as a drop-in replacement for standard objectives.

Limitations and future work. Our experiments demonstrate consistent gains from LK losses across target models and draft architectures, and several directions remain open for further exploration of acceptance-oriented training. The present study focuses on a specific adaptive scheduler for the hybrid objective and uses a fixed exponential aggregation scheme across draft heads. Future work should explore alternative scheduler parameterizations, learnable or datadependent per-head aggregation strategies, as well as the sensitivity of the adaptive schedule to the order of aggregation. Ablating these alternatives would further clarify the internal mechanisms behind the hybrid objective.

Our evaluation isolates the effect of the training objective under chain sampling and the standard sequential acceptance logic. While improvements in per-position acceptance rates are expected to transfer to more complex inference schemes, systematically verifying this is a natural next step. Evaluating LK-trained speculators under different draft construction strategies, such as tree-based sampling, and under alternative verification or acceptance mechanisms, such as block verification, would broaden the picture.

It would also be valuable to study the trade-off between training-data diversity and epoch count. Our setup uses a large generated corpus with a fixed training schedule, but the relative benefits of broader data coverage versus repeated optimization over fewer samples are an open question. Finally, direct optimization of the draft acceptance length τ, is another promising direction for aligning the objective even more closely with practical speculative decoding speedups.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically improving the computational efficiency of large language model inference through better training objectives for speculative decoding. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Amari, S.-i. and Nagaoka, H. Methods of Information Geometry, volume 191 of Translations of Mathematical Monographs. American Mathematical Society and Oxford University Press, Providence, RI, 2000.

Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T. MEDUSA: Simple LLM inference acceleration framework with multiple decoding heads. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of ICML’24, pp. 5209–5235. PMLR, 2024.

Cai, Y., Liang, X., Wang, X., Ma, J., Liang, H., Luo, J., Zuo, X., Duan, L., Yin, Y., and Chen, X. FastMTP: Accelerating LLM inference with enhanced multi-token prediction. arXiv preprint arXiv:2509.18362, 2025.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman,

- G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf,
- H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Hu, Y., Guo, J., Feng, X., and Zhao, T. AdaSPEC: Selective knowledge distillation for efficient speculative decoders. arXiv preprint arXiv:2510.19779, 2025.

Ji, H., Ke, P., Hu, Z., Zhang, R., and Huang, M. Tailoring language generation models under total variation distance. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=VELL0PlWfc.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th ACM Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626. ACM, 2023.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Li, J., Du, L., Zhao, H., Zhang, B.-w., Wang, L., Gao, B., Liu, G., and Lin, Y. Infinity Instruct: Scaling instruction selection and synthesis to enhance language models. arXiv preprint arXiv:2506.11116, 2025a.

Li, Y., Wei, F., Zhang, C., and Zhang, H. EAGLE: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning, 2024.

Li, Y., Wei, F., Zhang, C., and Zhang, H. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Advances in Neural Information Processing Systems, 2025b.

Liu, X., Yu, J., Park, J., Stoica, I., and Cheung, A. Speculative decoding: Performance or illusion? arXiv preprint arXiv:2601.11580, 2026.

Sandler, J., Christopher, J. K., Hartvigsen, T., and Fioretto, F. SpecDiff-2: Scaling diffusion drafter alignment for faster speculative decoding. arXiv preprint arXiv:2511.00606, 2025.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1889–1897. PMLR, 2015.

Wertheimer, D., Rosenkranz, J., Parnell, T., Suneja, S., Ranganathan, P., Ganti, R., and Srivatsa, M. Accelerating production LLMs with combined token/embedding speculators. arXiv preprint arXiv:2404.19124, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Wang, B., Li, B., Wang, C., Yu, D., Huang, F., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yin, M., Chen, M., Huang, K., and Wang, M. A theoretical perspective for speculative decoding algorithm. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024. ISBN 9798331314385.

Zhao, W., Pan, T., Han, X., Zhang, Y., Sun, A., Huang, Y., Zhang, K., Zhao, W., Li, Y., Zhou, J., Zhou, H., Wang, J., Liu, Z., and Sun, M. FR-spec: Accelerating large-vocabulary language models via frequencyranked speculative sampling. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3909–3921, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-2510. doi: 10.18653/v1/2025.acl-long.198. URL https: //aclanthology.org/2025.acl-long.198/.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging LLM-as-ajudge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, 2023.

Zhou, Y., Lyu, K., Rawat, A. S., Menon, A., Rostamizadeh, A., Kumar, S., Kagy, J.-F., and Agarwal, R. DistillSpec: Improving speculative decoding via knowledge distillation. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=rsY6J3ZaTF.

### A. Gradient Derivations

This appendix provides complete derivations of the gradients presented in Section 4. Throughout, we consider distributions p (target, fixed) and q = softmax(zq) (draft), where we optimize the logits zq.

- A.1. Softmax Jacobian The fundamental building block is the Jacobian of the softmax function:

∂qi ∂zq,j

= qi(δij − qj),

where δij is the Kronecker delta.

- A.2. KL Divergence Gradient The forward KL divergence is

KL(p∥q) =

i

pi log

pi qi

=

i

pi log pi −

i

pi log qi.

Only the second term depends on zq. Taking derivatives:

∂ KL ∂zq,j

= −

i

pi qi ·

∂qi ∂zq,j

= −

i

pi qi · qi(δij − qj)

= −

i

pi(δij − qj)

= −pj + qj

i

pi

= qj − pj. Thus,

|∇zq<br><br>KL(p∥q) = q − p.|
|---|

- A.3. Total Variation Distance Gradient

The TV distance is

- 1

- 2 i |pi − qi|.

TV(p,q) =

Define si = sign(qi − pi), then ∂|p

i−qi|

∂qi = si, and:

∂ TV ∂zq,j

- 1

- 2 i

si · qi(δij − qj)

=

- 1

- 2

sjqj − qj

=

siqi

i

- 1

- 2

qj (sj − Eq[s]). Thus,

=

|∇zq<br><br>TV(p,q) =<br><br>1<br><br>2<br><br><br>q ⊙ (s − Eq[s]),|
|---|

where ⊙ denotes elementwise multiplication.

##### A.4. Negative Log Acceptance Rate Gradient

For the loss LαLK = −log α:

∂LαLK ∂zq,j

1 α

∂α ∂zq,j

= −

1 α

∂ TV ∂zq,j

=

or in vector form:

|∇zqLαLK =<br><br>1 α∇zq<br><br>TV(p,q).|
|---|

This key relationship shows that optimizing −log α is equivalent to optimizing TV with an adaptive learning rate that scales inversely with acceptance rate.

##### A.5. Gradient Magnitude Analysis

To understand gradient behavior in practice, we analyze a representative regime that captures early-stage draft model training. Let V denote the size of vocabulary V and let S ⊂ V denote the support set – the tokens with non-negligible probability under p. Consider a draft distribution q that is approximately uniform (qi ≈ 1/V ), representing a randomly initialized model, while the target p is concentrated on |S| = k ≪ V tokens (pi ≈ 1/k for i ∈ S, and pi ≈ 0 otherwise). This regime is relevant because target LLM distributions are typically peaked on few plausible tokens, while undertrained drafts spread mass across the vocabulary.

KL gradient magnitude. For i ∈ S: (q − p)i ≈ 1/V − 1/k ≈ −1/k. For i ∈/ S: (q − p)i ≈ 1/V .

1 k2

1 V 2

1 k

1 V ≈

1 k

∥q − p∥2 ≈ k ·

+ V ·

=

+

.

√

Thus ∥∇KL∥ = O(1/

k).

TV gradient magnitude. In this regime, si = −1 for i ∈ S (since qi < pi) and si = +1 for i ∈/ S. Thus Eq[s] ≈ 1−2k/V . For i ∈ S: (si − Eq[s])qi ≈ −2/V . For i ∈/ S: (si − Eq[s])qi ≈ 2k/V 2 ≈ 0.

1 4 · k ·

4 V 2

k V 2

∥∇TV ∥2 ≈

=

.

√

Thus ∥∇TV ∥ = O(

k/V ), which vanishes for large V . LαLK gradient magnitude. In this regime, α ≈ k/V , so:

√

1 α∥∇TV ∥ ≈

V k ·

k V

1 √

∥∇LαLK∥ =

=

.

k

√

The 1/α factor resolves TV’s vanishing gradient problem, restoring O(1/

k) magnitude in the diffuse-q regime, while directly targeting acceptance rate.

- Table 3 summarizes the gradient components in each region.

Table 3. Gradient components for different losses in the diffuse-q, concentrated-p regime.

Loss Gradient on S Gradient off S KL −1/k +1/V TV −1/V ≈ 0 LαLK −1/k +1/V

### B. Connection to Negative Log-Likelihood

We establish a relationship between LαLK and the standard negative log-likelihood (NLL) used in language model training. When the target distribution p is a point mass at token x∗, i.e., p(x∗) = 1 and p(x) = 0 for x ̸= x∗, the acceptance rate simplifies to:

min(pi,qi) = min(1,q(x∗)) = q(x∗). Therefore:

α =

i

LαLK(p,q) = −log q(x∗), which is precisely the negative log-likelihood of x∗ under q.

#### C. Acceptance Rate as Densities Overlap Indeed, if we generalize (1) to continuous distributions we get

∞

- p(x)

- q(x)

α = Ex∼q min 1,

min(q(x),p(x))dx,

=

−∞

which is exactly the total area under the minimum of both density curves.

### D. Rejection Sampling with Greedy Draft Tokens

The standard speculative decoding algorithm (Leviathan et al., 2023) samples draft tokens from the draft distribution q and accepts them with probability min(1,p(x)/q(x)), where p is the target distribution. At non-zero temperatures, proper rejection sampling requires both the numerator p(x) and denominator q(x) to reflect the actual sampling distributions. However, the current vLLM implementation samples draft tokens greedily while still using temperature-scaled target logits in the acceptance criterion.

Under greedy sampling, the draft always selects x∗ = arg maxx q(x), substituting q(x∗) = 1 in the acceptance criterion. The acceptance probability becomes:

p(x∗) 1

= p(x∗).

αgreedy = min 1,

When the target distribution is confident and agrees with the draft (high p(x∗)), this works well. However, when the target distribution is diffuse, or we are in high-temperature sampling scenario, p(x∗) is small even if the draft correctly identifies the most likely token, leading to systematically low acceptance rates.

Since our LK losses directly optimize the true acceptance rate α = x min(p(x),q(x)) under temperature = 1 evaluating with greedy draft sampling introduces a mismatch between training and evaluation objectives. Our vLLM patch ensures that

evaluation faithfully measures the quantity we optimize during training.

### E. Draft Model Architecture Details

For dense target models (Llama-3.1-8B, Llama-3.3-70B), EAGLE-3 draft heads consist of a single transformer layer that mirrors the target model’s architecture and processes the concatenation of token embeddings and aggregated hidden states from the target model’s intermediate layers. For MoE target models (gpt-oss-20b, gpt-oss-120b, Qwen3-235B), we use a single dense transformer block rather than an MoE block. The intermediate dimension of the feed-forward network is chosen as

dffn = num experts per tok × dexpert,

where num experts per tok is the number of experts activated per token and dexpert is the intermediate dimension of each expert’s FFN. For DeepSeek-V3, we fine-tune the native Multi-Token Prediction (MTP) module, maintaining its original architecture.

MLP Speculator and MEDUSA use simpler architectures: both employ an MLP layer for each head. MEDUSA heads predict all positions in parallel from the same hidden state without token-level autoregression. Unlike EAGLE-3, which shares weights across positions, MLP speculator and MEDUSA train fully independent heads for each speculative position.

### F. Full Experimental Results

- Table 4. Average acceptance length τ and end-to-end speedup relative to the baseline without speculative decoding. We evaluate speculative decoding methods trained with different objectives on MT-Bench, HumanEval, and GSM8K in a low-latency setting with batch size 1, and compare them against public Hugging Face checkpoints at temperatures T =0 and T =1. Type: Ours = models trained from scratch with the specified objective; HF = public HuggingFace checkpoints evaluated with the same inference pipeline.6

Model Method Type Setup Temperature = 0 Temperature = 1 MT-Bench HumanEval GSM8K MT-Bench HumanEval GSM8K

τ / speedup τ / speedup τ / speedup τ / speedup τ / speedup τ / speedup LLaMA 3.1 EAGLE-3 Ours KL 3.75 / 2.60 4.82 / 3.30 4.50 / 3.05 3.39 / 2.29 4.31 / 3.00 3.88 / 2.63 8B Instruct TV 2.81 / 1.96 3.42 / 2.39 3.34 / 2.28 2.67 / 1.83 3.25 / 2.26 3.12 / 2.11

LαLK 3.77 / 2.61 4.82 / 3.28 4.55 / 3.07 3.50 / 2.30 4.48 / 3.03 3.98 / 2.72 LLK, λ = 0.5 3.78 / 2.63 4.84 / 3.32 4.53 / 3.07 3.35 / 2.36 4.36 / 3.03 3.95 / 2.68 LλLK, η = 0.7 3.79 / 2.61 4.88 / 3.31 4.54 / 3.07 3.53 / 2.37 4.45 / 2.98 3.98 / 2.66 LλLK, η = 1 3.80 / 2.64 4.83 / 3.33 4.54 / 3.08 3.51 / 2.30 4.47 / 3.01 3.96 / 2.70 LλLK, η = 3 3.84 / 2.62 4.89 / 3.35 4.57 / 3.10 3.48 / 2.39 4.52 / 3.08 4.02 / 2.72 LλLK, η = 10 3.67 / 2.53 4.85 / 3.33 4.53 / 3.06 3.34 / 2.30 4.51 / 3.01 4.03 / 2.72

HF RH-8B 3.08 / 2.09 3.90 / 2.65 3.57 / 2.37 2.67 / 1.83 3.36 / 2.31 3.08 / 2.05 YH-8B 3.44 / 2.42 4.37 / 3.07 3.84 / 2.65 2.97 / 2.22 3.75 / 2.50 3.21 / 2.14 ZK-8B 3.54 / 2.45 4.49 / 3.08 3.97 / 2.71 3.06 / 2.68 3.96 / 2.82 3.36 / 2.28

LLaMA 3.3 EAGLE-3 Ours KL 4.01 / 3.01 5.18 / 3.87 5.16 / 3.78 3.76 / 2.82 4.86 / 3.66 4.89 / 3.61 70B Instruct LαLK 3.95 / 3.00 5.12 / 3.82 5.13 / 3.78 3.76 / 2.85 4.94 / 3.71 4.91 / 3.66

LλLK, η = 3 4.00 / 2.98 5.21 / 3.87 5.21 / 3.83 3.89 / 2.89 5.08 / 3.77 5.01 / 3.69 HF RH-70B 3.11 / 2.35 3.99 / 2.99 3.62 / 2.68 2.88 / 2.10 3.62 / 2.69 3.29 / 2.44

YH-70B 3.13 / 2.50 3.96 / 3.12 3.76 / 2.90 2.77 / 2.07 3.49 / 2.63 3.34 / 2.50 GPT-OSS EAGLE-3 Ours KL 3.31 / 1.41 3.08 / 1.32 4.01 / 1.63 3.12 / 1.30 2.89 / 1.20 3.51 / 1.44 20B LαLK 3.27 / 1.41 3.07 / 1.32 4.03 / 1.63 3.28 / 1.35 2.97 / 1.25 3.65 / 1.49

###### LλLK, η = 3 3.35 / 1.42 3.11 / 1.34 4.00 / 1.63 3.20 / 1.35 3.01 / 1.26 3.65 / 1.49

HF RH-20B† 2.90 / 1.26 2.70 / 1.17 3.49 / 1.45 2.63 / 1.12 2.43 / 1.04 3.00 / 1.26 GPT-OSS EAGLE-3 Ours KL 2.81 / 1.57 2.47 / 1.54 3.00 / 1.77 2.53 / 1.20 2.27 / 1.19 2.57 / 1.32 120B LαLK 2.80 / 1.59 2.51 / 1.57 3.08 / 1.82 2.67 / 1.49 2.39 / 1.45 2.78 / 1.63

LλLK, η = 3 2.86 / 1.62 2.51 / 1.60 3.06 / 1.82 2.69 / 1.48 2.44 / 1.48 2.81 / 1.66 Qwen 3 EAGLE-3 Ours KL 3.33 / 2.09 4.65 / 2.73 4.76 / 2.88 2.96 / 1.84 4.09 / 2.37 4.27 / 2.50 235B A22B LαLK 3.29 / 2.10 4.68 / 2.77 4.82 / 2.91 3.11 / 1.96 4.31 / 2.46 4.50 / 2.61 Instruct LλLK, η = 3 3.36 / 2.13 4.74 / 2.78 4.87 / 2.93 3.18 / 1.98 4.42 / 2.52 4.65 / 2.68

HF RH-235B 2.92 / 1.85 4.06 / 2.42 4.31 / 2.63 2.56 / 1.61 3.52 / 2.06 3.78 / 2.22

ZK-235B‡ 3.06 / 1.94 4.54 / 2.70 4.90 / 2.94 2.64 / 1.63 4.03 / 2.36 4.43 / 2.57 DeepSeek MTP Ours KL 3.96 / 2.22 4.74 / 2.62 5.67 / 2.95 3.66 / 1.85 4.39 / 2.17 5.23 / 2.30 V3 0324 LλLK 4.00 / 2.22 4.77 / 2.64 5.72 / 2.97 3.88 / 2.14 4.64 / 2.42 5.51 / 2.71

HF DS-685B 2.90 / 1.66 3.28 / 1.84 3.41 / 1.82 2.82 / 1.54 3.17 / 1.59 3.27 / 1.47

† Results are based on the checkpoint revision available at the time experiments were conducted, prior to the subsequent weight update released during the review period. ‡ This checkpoint employs a wider MLP (24,576 vs. 12,288) and a smaller vocabulary (32k vs. 64k), yielding a slightly larger parameter count.

6RH=RedHatAI, YH=yuhuili, ZK=zhuyksir, DS=deepseek-ai. RH-8B=Llama-3.1-8B-Instruct-speculator.eagle3; YH-8B=EAGLE3-LLaMA3.1-Instruct-8B; ZK-8B=EAGLE3-Llama-3.1-8B-Instruct; RH-70B=Llama-3.3-70B-Instruct-speculator.eagle3; YH-70B=EAGLE3-LLaMA3.3-Instruct-70B; RH-20B=gpt-oss-20b-speculator.eagle3; RH-235B=Qwen3-235B-A22B-Instruct-2507-speculator.eagle3; ZK-235B=EAGLE3-Qwen3-235B-A22B-Instruct-2507-FP8; DS-685B=DeepSeek-V3-0324.

