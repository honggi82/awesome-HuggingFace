## Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO

Yiming Ren*12 Yiran Xu*1 Zicheng Lin*1 Chufan Shi1 Yukang Chen3 Dingdong Wang3 Tianhe Wu4 Jujie Wang1 Yujiu Yang1 Yu Qiao2 Ruihang Chu1

# arXiv:2605.30789v2[cs.LG]2Jun2026

### Abstract

We identify a new dimension for enhancing rollout diversity in Group Relative Policy Optimization (GRPO) for LLMs. While GRPO relies on diverse rollouts, prevailing strategies primarily increase diversity by injecting more token-level randomness, which may introduce step-wise noise and lead to incoherent trajectories. We uncover that smaller models within the same model family inherently exhibit higher policy-level diversity, indicated by their superior pass@k relative to larger counterparts as sample counts increase. Unlike token-level noise, this diversity is temporally correlated, preserves logical consistency, and provides structured exploration signals for gradient estimation. We thus propose S2L-PO (Smallto-Large Policy Optimization), a framework that leverages fixed small models as natural explorers to train larger models. To balance exploration and exploitation, we design a progressive annealing strategy that transitions from offline small-model rollouts to the large learner’s own sampling. This shift elegantly avoids mid-training performance drops caused by the small model’s capacity limits, achieving faster convergence and unlocking a higher performance ceiling. S2L-PO improves accuracy on diverse mathematical reasoning benchmarks (e.g., +8.8% on AIME 24 using a 1.7B explorer to guide the 8B model) while reducing rollout compute.

### 1. Introduction

Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for improving the reasoning capabilities of large language models (Guo et al., 2025;

1Tsinghua University 2Shanghai AI Laboratory 3The Chinese University of Hong Kong 4City University of Hong Kong. Correspondence to: Yu Qiao <qiaoyu@pjlab.org.cn>, Ruihang Chu <ruihangchu@mail.tsinghua.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

Hong et al., 2024; Wang et al., 2025d). Group Relative Policy Optimization (GRPO) (Shao et al., 2024), in particular, has gained widespread adoption due to its simplicity and effectiveness: it samples multiple candidate solutions per prompt, computes group-relative advantages, and updates the policy without requiring a separate critic network. A key factor in GRPO’s success is the diversity of sampled rollouts. When candidates within a group are too homogeneous, the advantage signals collapse and learning stagnates (Gu et al., 2025; Wang et al., 2025d; Zhang et al., 2025b).

Prevailing strategies for increasing rollout diversity primarily operate on the token level. A common approach is temperature scaling, which raises the original sampling temperature to inject more randomness into individual token selection. Yet, high-temperature sampling can trigger entropy explosion (Nguyen et al., 2024; Shi et al., 2024b; Wang et al., 2025c; Yang et al., 2025b; Zhuang et al., 2025), where the policy explores indiscriminately across all tokens, leading to training instability and degraded reasoning performance. More critically, because elevated temperature adds randomness independently at each decoding step, small deviations compound over long reasoning chains, making it difficult to maintain a consistent logical flow. Such resulting rollouts may exhibit high surface diversity in terms of token entropy but often suffer from low behavioral coherence. Ultimately, this approach is less effective at providing the structured exploration signals that GRPO requires. While other works explore curating diverse response sets to improve training signals or rewarding intra-group diversity (Anschel et al., 2025; Chen et al., 2025), these strategies involve data engineering and extra computational overhead, limiting their scalability to new tasks without significant costs.

We present an empirical finding to explore an alternative dimension for enhancing diversity. When comparing models of different sizes on mathematical reasoning benchmarks, we observe a surprising pattern: while larger models outperform their smaller counterparts at pass@1, this gap shrinks and can even reverse as k increases (see Fig. 2). For instance, a 4B model surpasses an 8B model in pass@k once k ≥ 32, and it can also outperform a 14B model when the sample budget is sufficiently large (e.g., k ≈ 200). As smaller models have a lower performance floor, their competitiveness or advantage at higher sample counts suggests that they pos-

#### GRPO

KL

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

𝑟 𝑟 𝑟 𝑟

𝐴 𝐴 𝐴

𝑞 at training step 𝑖 𝑂 𝑂 𝑂

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Reference model

Group Compute

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Policy model

[Figure 21]

[Figure 22]

…

[Figure 23]

[Figure 24]

…

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

…

Reward model

[Figure 29]

[Figure 30]

[Figure 31]

𝐴

𝑂

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

#### S2L-PO

KL

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Large policy model

𝑟 𝑟 𝑟

𝐴 𝐴 𝐴

𝑞 at training step 𝑖 𝑂 𝑂 𝑂

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Reference model

Group Compute

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

…

…

[Figure 63]

[Figure 64]

[Figure 65]

Small policy model

…

Reward model

[Figure 66]

[Figure 67]

[Figure 68]

𝑟

𝐴

𝑂

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Training steps

- Figure 1. S2L-PO (Bottom) simply modifies the rollout generation process of standard GRPO (Top). Motivated by the observation that smaller models inherently exhibit higher policy-level diversity, S2L-PO leverages a frozen smaller policy model to sample diverse rollouts for training a larger model. In early training, rollouts are primarily sampled from the smaller model to encourage diverse exploration. As training progresses, sampling smoothly transitions through a mixture of smaller and larger models, and ultimately recovers standard on-policy GRPO to balance exploration and exploitation.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

AIME24AIME25

k k

temperature = 0.7, top-p = 0.8 temperature = 0.6, top-p = 0.95

[Figure 87]

- Figure 2. Pass@k curves on AIME24 and AIME25 for Qwen3 Base models of various scales. While larger models perform better at small k, smaller models continue to improve as k increases and can match or exceed larger models under large sampling size.

level compression induces a structured shift in the policy’s inductive bias. Unlike token-level diversity that perturbs the action distribution through step-wise noise along a reasoning trajectory, parameter-level compression applies a time-invariant perturbation to shift the entire policy. As analyzed in Sec. 3.1, this preserves temporal correlation and enhances internal consistency. It further prevents gradient dilution by focusing exploration on structured reasoning strategies rather than uncoordinated local flips, providing more informative updates for the learner. We summarize this distinction as follows: token-level randomness perturbs the action; parameter-level compression perturbs the policy.

Building on this insight, we propose S2L-PO (Small-toLarge Policy Optimization), a framework that leverages weaker small models as natural explorers to generate rollouts for training stronger large models (see Fig. 1). Since the small model can provide superior policy diversity per compute unit, we fix its parameters to generate rollouts offline. This setting avoids the instability of early-stage on-policy updates caused by mismatched model capacities and enables highly efficient parallelization of rollout generation. To balance exploration and exploitation, we design a progressive annealing strategy that transitions from small-model exploration to on-policy learning. Initially, the small model provides the entire or the majority of rollouts to maintain diverse exploration and prevent mode collapse. As training progresses, we gradually shift the sampling role to the large model to mitigate the distribution mismatch between the small sampler and the large learner. This approach effectively prevents mid-training performance degradation and ultimately achieves a higher ceiling. Since S2L-PO only modifies the rollout process, it remains seamlessly compatible with existing GRPO implementations.

sess an inherent diversity, stemming not from token-level randomness but from more varied solution strategies (Bansal et al., 2024; Dragoi et al., 2025; Yue et al., 2025).

We characterize this phenomenon as a form of policy-level diversity. Smaller models typically undergo distillation from larger models within the same family, ensuring distributional alignment while reducing parameter count. This parameter-

We comprehensively evaluate our approach across two model families (Qwen3 and InternLM2.5) on four mathematical reasoning benchmarks (AIME24, AIME25, MATH500, and OlympiadBench). Across various settings, smallto-large policy sampling consistently improves both final performance and sample efficiency over standard GRPO, reaching stronger Pass@1 with fewer effective training steps (e.g., using a 1.7B explorer to guide an 8B model yields an average gain of about 9%). On an out-of-domain benchmark (CommonsenseQA), our method matches or marginally improves over GRPO, suggesting that the benefits do not come at the expense of generalization. Code is available at https://github.com/ qishisuren123/S2L-PO.

### 2. Preliminary

##### 2.1. Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) is an on-policy policy-gradient method tailored to RLVR settings, where supervision is provided by verifiable rewards (e.g., rule-based correctness checks). GRPO optimizes a policy model πθ without training an explicit value function (critic). Instead, it estimates advantages via within-group relative comparisons among multiple samples generated for the same query, which reduces both compute and engineering overhead.

Formally, for each query q ∼ D, GRPO samples a group of k candidate outputs O = {o1,o2,...,ok} from the behavior policy πθ

and evaluates each output with a scalar reward r(oi). It then computes a group-relative advantage by standardizing rewards within the sampled group:

rollout

r(oi) − mean({r(oj)}kj=1) std({r(oj)}kj=1) + ϵadv

. (1)

Ai =

θ(oi|q)

Let ρi = π

πθrollout(oi|q) denote the importance sampling ratio. GRPO uses a PPO-style clipped surrogate objective with KL regularization toward a reference policy πref. We optimize JGRPO(θ) defined as:

k

1 k

E

min ρiAi, clip(ρi,1 − ϵclip,1 + ϵclip)Ai

i=1

−β DKL(πθ ∥πref) . (2)

globally incoherent trajectories. This motivates our exploration of policy-level perturbations as an alternative source of structured diversity.

##### 2.2. Distillation Introduces Perturbations

In this work, we study the parameter-count compression within a single model family and its implications for exploration in RL-style fine-tuning. Rather than viewing compression purely as an efficiency tool, we interpret the compression-and-distillation process as inducing a policylevel perturbation (Gu et al., 2024; Hinton et al., 2015; Park & Cho, 2025; Peng & Zhang, 2025). Although compression is often motivated by deployment constraints (e.g., memory, latency, and serving cost), the student is typically optimized to retain the teacher’s task behavior under a reduced parameter budget. As a result, the teacher-to-student mapping is not arbitrary: it induces a structured shift in inductive biases and decision boundaries. We leverage this property and treat the compressed student as a coherent deviation from its teacher in policy space, providing a source of exploration diversity.

Take Qwen3 dense series (Yang et al., 2025a) as example, which represents a currently strong and widely adopted base model family. For models at or below 14B parameters, they are obtained via a unified larger-to-smaller distillation in final stages. As reported, each model is trained as the student to align its logits to those of a larger teacher (e.g., Qwen3-32B or Qwen3-235B-A22B) by minimizing a KL-divergence objective during on-policy distillation. This yields a controlled compression setting: students across scales share a consistent distillation procedure and teacher family, ensuring behavioral proximity while allowing capacity reduction to induce meaningful, structured deviations.

Formally, let πθ⋆ denote the teacher policy and {πθ

k}k∈K denote student policies at different parameter scales within the same series, where K = {1.7B,4B,8B,14B}. Since πθ

is trained to approximate πθ⋆ under distillation, we model compression as an effective perturbation in parameter space, formulated as

k

###### θk ≈ θ⋆ + δθ,k, (3)

where δθ,k captures the structured change induced by compression and distillation. Equivalently, this corresponds to a controlled deviation in policy space:

Since GRPO relies on within-group relative rewards, its gradient quality is sensitive to the diversity of sampled candidates. When candidates are homogeneous, advantage signals vanish and learning stagnates. The standard remedy, temperature scaling, injects token-level noise that is temporally independent, often yielding locally random but

(· | q) ≈ πθ⋆(· | q) + ∆π,k(· | q), (4)

πθ

k

where ∆π,k is a coherent shift arising from reduced capacity, rather than a token-level random perturbation. In this paper, we validate the perturbation view on both Qwen3 (Yang et al., 2025a) and InternLM2.5 (Cai et al., 2024) families.

### 3. Method

Given that compression induces structured, temporally consistent perturbations, we first analyze why policy-level perturbations yield new kinds of exploration signals compared to token-level noise (Section 3.1). Then we present S2LPO framework that leverages this property (Section 3.2).

##### 3.1. Token-Level vs. Policy-Level Perturbations

Given a policy and a query, GRPO samples a group of rollouts and constructs group-relative advantages for policy updates. The exploration mechanism determines how these rollouts deviate from each other in policy space and directly influences the quality of gradient estimates. In standard GRPO implementations, rollouts are sampled with a modest non-zero temperature to balance training stability and within-group diversity, which already introduces a baseline level of token-level randomness. In this paper, we treat this default temperature as part of the GRPO baseline and focus on additional sources of diversity beyond it.

the considered horizon, i.e.,

p ≤ Pr aj ̸= a⋆j | Mj−1 = 1 j ∈ {1,...,t}, (9)

Pr(Mt = 1) ≤ (1 − p)t, (10) so the mass of trajectories that share a common early prefix decays exponentially with t.

This decay implies that for long-horizon outputs, late tokens are increasingly generated under a mixture over divergent prefixes. A convenient proxy for the resulting loss of temporal dependence is the growth of Pr(Mt = 0). In particular, for bounded features f(at) and g(as) with ∥f∥∞,∥g∥∞ ≤ 1 and t < s, one can obtain under a mild mixture/coupling assumption a problem-dependent constant C > 0 such that

Cov(f(at),g(as) | q) ≤ C Pr(Mt = 0). (11)

Thus, as Pr(Mt = 0) grows with t for long generations, long-range cross-token dependence weakens, making earlier and later decisions less mutually consistent.

Token-level perturbations. It refers to introducing additional step-wise randomness in action selection beyond the baseline sampling temperature used in GRPO. A typical instance is sampling from a softened distribution

exp(lt(a)/T) a′ exp(lt(a′)/T)

at ∼ πtok(· | st), πtok(a | st) =

,

(5) where lt(·) denotes the logits at step t and T controls the perturbation strength. Equivalently, this process can be expressed using the Gumbel–Max formulation

lt(a)/T + ϵt(a) , {ϵt(·)}t≥1 i.i.d., (6)

at = arg max

a

where the noise sources are independent across steps. Importantly, while the injected noise sources are i.i.d., the realized tokens {at} are generally not i.i.d. because the state st depends on previous actions.

Token-level diversification draws actions from a perturbed conditional distribution at each step, at ∼ πtok(· | st). Let o = (a1,...,aL) be the resulting sequence and define the prefix match event

Mt := I{(a1,...,at) = (a⋆1,...,a⋆t)}, (7)

where o⋆ denotes a deterministic reference decoding trace under the base policy, used only to define whether a rollout remains on the same decision path. For any step t,

Pr(Mt = 1) =

t

Pr aj = a⋆j | Mj−1 = 1 . (8)

j=1

Moreover, consider a regime where token-level randomness is increased relative to the GRPO baseline, so that the perstep deviation probability admits a lower bound p > 0 over

Policy-level perturbations via parameter-level compression. In contrast, parameter-level compression (in this work, primarily via distillation to a smaller model) induces an effective structured perturbation in parameter space. We abstract this effect by an equivalent additive perturbation

θ˜ = θ + δθ, at ∼ πθ˜(· | st), (12)

where δθ represents a time-invariant modification of the policy parameters during the rollout. Although the resulting logit shifts depend on context through the forward pass, all steps share the same perturbed policy πθ+δ

. For any fixed state s, define the local distributional shift

θ

(a | s) − πθ(a | s). (13)

∆πs(a) := πθ+δ

θ

Since the same δθ is applied at every step, the induced shifts {∆πs

t}Lt=1 are coupled through shared parameters, yielding trajectory-level deviations that are typically temporally correlated rather than independent across t. Intuitively, this correlation encourages trajectories to follow a coherent alternative strategy throughout the rollout generation.

Implications for gradient estimation in GRPO. We now examine how these differences affect policy-gradient estimates. For a sampled trajectory oi with group-relative advantage

ri − µr σr

, (14) and the GRPO policy-gradient contribution is

Ai =

gi = Ai∇θ log πθ(oi | q) = Ai

L

∇θ log πθ(ai,t | si,t).

t=1

(15)

Parameter-level compression

High temperature sampling

Output A

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

[Figure 88]

Start

- Output B

- Output C

- Output D

[Figure 89]

Start

Output

[Figure 90]

[Figure 91]

Decoding steps Decoding steps

(a) Token-level perturbation

(b) Policy-level perturbation (c) Performance comparison

- Figure 3. Two ways to increase rollout diversity under standard GRPO. (a) Increasing token-level perturbation (e.g., higher sampling temperature) introduces step-wise stochasticity that accumulates over decoding steps, often reducing long-range coherence. (b) Policylevel perturbations (e.g., parameter-level compression within a model family) induce temporally consistent trajectory deviations, yielding diverse yet structured policy paths. (c) Increasing token-level randomness beyond the GRPO baseline yields limited gains, whereas adding policy-level diversity enables more effective exploration and significantly better results. Blue thermometers indicate the default GRPO temperature; red indicate higher temperatures. Different colors denote token-level diversity; different shapes denote policy-level diversity.

Structured gradients under policy-level perturbations. For a fixed parameter-level perturbation δθ shared across the rollout, a local expansion yields, for any fixed trajectory,

Let ui,t := ∇θ log πθ(ai,t | si,t) denote the per-step score function, so gi = Ai Lt=1 ui,t. Since Ai is a scalar shared across steps, it scales squared norms by A2i but does not change the cross-step interference mechanism; we therefore analyze Lt=1 ui,t for clarity.

(o | q) ≈ ∇θ log πθ(o | q)+∇2θ log πθ(o | q)δθ.

∇θ log πθ+δ

θ

(19)

Gradient interference under token-level perturbations. Expanding the squared norm gives

Although Eq. (19) is local, it highlights a key qualitative difference: the rollout is generated under a single, consistently shifted policy, which tends to induce temporally consistent deviations throughout the trajectory. As a result, per-step score contributions are more likely to remain aligned across time, increasing cross-step reinforcement in Eq. (16) and mitigating the long-lag cancellation behavior implicit in Eq. (18). In long-horizon settings, this yields a more coherent trajectory-level gradient signal for GRPO group comparisons.

L

L

2

∥ui,t∥2 + 2

⟨ui,t,ui,s⟩. (16)

ui,t

=

t=1

t=1

1≤t<s≤L

Under strengthened token-level perturbations, prefix divergence suppresses long-range dependence (cf. (11)). Concretely, given bounded scalar projections zi,t := ⟨ui,t,v⟩ with ∥v∥ = 1, there exist tasks such that

More formally, we show in Proposition E.4 (Appendix E) that the cross-step covariance under policy-level perturbation admits a positive lower bound:

Cov(zi,t,zi,s | q) ≤ c Pr(Mt = 0), t < s, (17)

where c > 0 is a constant depending on the bound of zi,t. A formal proof via the law of total covariance, which eliminates the need of coupling assumptions and yields c = 5B2 with B bounding |zi,t|, is given in Proposition E.2 (Appendix E). As Pr(Mt = 0) grows with t in long outputs, correlations between distant steps are suppressed. Consequently, the large-lag cross terms in Eq. (16) are less coherent and tend to cancel in expectation, so the accumulation behaves closer to a random-walk sum when long-range alignments vanish:

|Cov(˜zi,t,z˜i,s | q)| ≥ γ − 5B2 Pr std

(Mt=0) − O(∥δθ∥3),

(20) where γ := E[v⊤HtΣδHs⊤v | q] > 0 captures Hessian alignment under the parameter perturbation δθ, and Prstd is evaluated at the standard (unperturbed) temperature. Unlike the token-level upper bound in Eq. (17) that becomes vacuous as Pr(Mt=0) → 1, this lower bound remains positive when the Hessian alignment γ is sufficiently large, ensuring constructive gradient interference across steps.

Takeaway. Token-level randomness can accumulate over decoding steps, which may break long-range coherence and increase gradient interference as shown in Eqs. (16)–(18); policy-level perturbations induce time-correlated deviations that preserve coherence and yield more structured GRPO gradients. Fig. 3 illustrates these two different ways by showcasing their mechanisms, as well as their actual impact on model performance.

L

L

2

E ∥ui,t∥2 (18)

E

≈

ui,t

t=1

t=1

For long-horizon reasoning (e.g., mathematical solution generation), this implies weaker cross-step reinforcement: per-token score contributions are less consistently aligned across the horizon, which can make the trajectory-level update direction noisier and less stable under GRPO.

- 3.2. S2L-PO: Small-to-Large Policy Optimization

Guided by the above analysis, we propose S2L-PO, a framework that leverages smaller (compressed) models for exploration while training larger models for exploitation. The core idea is simple: since smaller models provide richer behavioral diversity per unit compute, we use them to generate rollouts and train the larger policy via GRPO. The complete procedure is summarized in Algorithm 1.

Mixed rollout generation. At each training step, we construct a mixed rollout distribution. Given a group size G, we sample Gw candidates from a frozen smaller policy πω and Gs = G − Gw candidates from the trainable larger policy πθ. The smaller policy remains frozen throughout training and serves solely as an exploration agent. The larger policy is updated using GRPO with group-relative advantages computed over the combined candidate set.

Progressive annealing. We linearly anneal the smallerto-larger ratio over the first Tmix training steps. In our implementation, Tmix defaults to the first half of the total training process. Early in training, when the larger policy is unstable and prone to mode collapse, the smaller model provides diverse exploration at low cost, alleviating vanishing advantage signals. As training converges, reducing Gw mitigates distribution mismatch, ensuring the final policy is optimized under its own behavior. After step Tmix, the framework recovers standard on-policy GRPO.

Compatibility and efficiency. S2L-PO does not modify the GRPO objective, advantage construction, or optimization procedure; it only changes how rollouts are generated. As a result, it can be plugged into existing GRPO pipelines with minimal engineering effort and remains orthogonal to complementary techniques such as reward shaping or curriculum learning. In addition, using a smaller rollout policy reduces the per-sample generation cost, and the same weakmodel rollouts can be reused across multiple strong-model training runs, further amortizing rollout compute. Since rollout generation is typically the dominant time and compute bottleneck in GRPO, these properties translate into direct savings in FLOPs and wall-clock time, and in principle can shorten end-to-end training by reducing the rollout burden.

- 4. Experiment

- 4.1. Experiment Settings

We train on the deduplicated DAPO17k (Yu et al., 2025) focusing on verifiable multi-step reasoning. For evaluation we choose four mathematical reasoning benchmarks: AIME 2024, AIME 2025 (Balunovi´c et al., 2025), MATH500 (Hendrycks et al., 2021), and OlympiadBench (He et al., 2024), and additionally report out-of-domain (OOD) gen-

Algorithm 1 S2L-PO: GRPO with Progressive smaller-tolarger Rollout Sampling

Require: Trainable policy πθ, frozen smaller policy πω, reward function rϕ, prompt dataset D, group size G, total training steps T, transition step Tmix, GRPO update steps per iteration U

Ensure: Optimized policy πθ

- 1: for i = 1 to T do
- 2: Sample a batch of prompts Db ⊂ D.
- 3: if i ≤ Tmix then
- 4: {Progressive smaller-to-larger rollout phase}
- 5: α ← 1 − T i−1

mix−1

- 6: Gw ← ⌈αG⌉, Gs ← G − Gw
- 7: else
- 8: {Pure on-policy GRPO phase}
- 9: Gw ← 0, Gs ← G
- 10: end if
- 11: for all q ∈ Db do
- 12: if Gw > 0 then
- 13: Sample Gw candidates from πω(· | q)
- 14: end if
- 15: Sample Gs candidates from πθ(· | q)
- 16: Form candidate group O(q)
- 17: end for
- 18: Compute rewards using rϕ and group-relative advantages following GRPO
- 19: for u = 1 to U do
- 20: Update θ by maximizing the GRPO objective
- 21: end for
- 22: end for
- 23: return πθ

eralization on CommonsenseQA (Talmor et al., 2019). All evaluations are in nothink mode following the Qwen3 technical report (Yang et al., 2025a). We sample 16 rollouts per question and compute Pass@1 by averaging the perproblem success indicator over the dataset. To demonstrate cross-family generalizability, we evaluate on two model families: Qwen3-Base (Yang et al., 2025a) and InternLM2.5Base (Cai et al., 2024). For Qwen3, the 1.7B and 4B variants serve as smaller rollout actors for 8B and 14B target policies. For InternLM2.5, the 1.8B model serves as explorer for the 7B target. All runs are conducted on a single node with 8 NVIDIA L20 GPUs using the default GRPO configuration in verl (Sheng et al., 2024).

##### 4.2. Main Results

Small-to-large sampling improves both convergence speed and final performance. As illustrated in Fig. 3a and Fig. 3b, our approach leverages a smaller model to introduce policy-level diversity. Fig. 3c contrasts this with increasing token-level noise (Temperature = 1.5). Unlike high-temperature sampling, which suffers from instability

[Figure 92]

- Table 1. Cross-family main results (Pass@1, %). We evaluate S2LPO across two model families (Qwen3 and InternLM2.5) on four benchmarks. ∆ denotes improvement over the GRPO baseline.

Family Method AIME24 AIME25 MATH-500 OlympiadBench

Qwen3

8B GRPO 15.0 12.1 57.3 18.1 1.7B→8B S2L 23.8 22.5 61.5 19.7 ∆ +8.8 +10.4 +4.2 +1.7

14B GRPO 18.0 12.9 58.7 18.9 4B→14B S2L 24.4 14.6 62.7 21.9 ∆ +6.4 +1.7 +4.0 +3.0

InternLM 2.5

7B GRPO 0.1 0.1 18.6 2.2 1.8B→7B S2L 4.6 3.5 22.6 3.4 ∆ +4.5 +3.4 +4.0 +1.2

- Table 2. Out-of-domain evaluation on CommonsenseQA. Accuracy (%) of strong models trained on math data and evaluated on CommonsenseQA without additional tuning.

[Figure 93]

Qwen3-8B-Base Qwen3-14B-Base

GRPO S2L-PO-1.7B S2L-PO-4B GRPO S2L-PO-4B CommonsenseQA 63.9 64.2 67.8 67.2 70.7

Figure 4. S2L-PO improves both final performance and convergence speed. Pass@1 on AIME24&25 versus effective training progress for different scale transitions. S2L-PO uses a smaller model to generate part of each rollout group early in training and progressively anneals to fully on-policy GRPO.

AIME24 with K =64 rollouts: Self-BLEU (to reflect text repetition), Edit Diversity (to reflect token-level difference), and Unique Answer Ratio (to reflect proportion of distinct final answers). As shown in Table 3, all three metrics are monotonic with model size. The 1.7B model achieves 21% higher Unique Answer Ratio than 14B (0.576 vs. 0.476), confirming genuine strategy-level diversity.

and regresses to significantly lower Pass@1 in later stages, our policy perturbation proves to be more stable, converges faster, and yields superior results. In Fig. 4 and Table 1, we further observe that our method consistently reaches a higher performance ceiling than standard GRPO. For example, in the Qwen3-8B-Base setting, using a 1.7B explorer improves performance by approximately 9% compared to the baseline. The initial boost from the smaller model’s diversity builds a stronger foundation, allowing the larger model to stabilize at this superior level. Notably, this improvement comes with reduced computational overhead. By offloading a portion of rollout generation to a smaller model and allowing for the reuse of these off-policy trajectories, we significantly reduce the total training FLOPs. As shown in Table 1, S2L-PO achieves consistent improvements across two model families (Qwen3 and InternLM2.5) on four benchmarks with varying scale configurations. As illustrated in Table 2, our method outperforms standard GRPO on CommonsenseQA (OOD). Specifically, for Qwen3-8B-Base, the S2L-PO-4B variant achieves an accuracy of 67.8% compared to 63.9% for the vanilla baseline, indicating that the diverse exploration preserves the model’s general reasoning capabilities and improves robustness. We further extended this to Qwen3-14B-Base using S2L-PO-4B, observing similar gains over the vanilla GRPO baseline.

Controlled experiment on rollout diversity. We filter out diverse rollouts from the small model so its diversity metrics match the large model. As shown in Table 4, performance drops back to the GRPO baseline, demonstrating that S2LPO’s gains are driven by the small model’s policy-level diversity, not by other factors such as off-policy mixing.

##### 4.4. Ablation Study

Pure small-model rollouts are not sufficient for sustained performance gains. Given the superior exploration capability of small models demonstrated above, a natural question arises: can we rely exclusively on small-model rollouts throughout training? Fig. 5 addresses this by evaluating a “small-only” baseline (Chen et al., 2026; Wang et al., 2025b) that never transitions to the standard GRPO. Initially, this baseline exhibits rapid performance gains, outpacing the vanilla GRPO. However, this advantage is transient: as training progresses, performance plateaus and eventually regresses, failing to reach the peak performance achieved by our progressive annealing method. We attribute this to the widening distribution shift between the static small-model explorer and the evolving large-model learner.

##### 4.3. Diversity Analysis

Quantitative measurement of policy-level diversity. To validate that S2L-PO’s gains stem from policy-level diversity, we design three complementary metrics measured on

Progressive transition vs. abrupt switch: gradual handover is strictly better. Having established the need for a transition, we further investigate how this handover should

- Table 3. Diversity metrics across model scales on AIME24 (K = 64). All metrics are strictly monotonic: smaller models are more diverse.

Model Self-BLEU ↓ Edit Div. ↑ Unique Ans. ↑

1.7B 0.314 0.788 0.576 4B 0.334 0.773 0.523 8B 0.336 0.769 0.492 14B 0.352 0.760 0.476

- Table 4. Controlled experiment: removing diversity from the small model’s rollouts eliminates S2L-PO’s advantage.

Config AIME24 AIME25

S2L-PO (1.7B→8B) 23.8 22.5 S2L-PO (w/o diversity) 14.7 12.0 GRPO Baseline 15.0 12.1

be executed. We compare our progressive annealing (linear decay) against an abrupt two-phase switch. Fig. 6 shows that the progressive transition consistently outperforms the abrupt switch. The abrupt strategy introduces a sharp shock to the training distribution, causing instability as the model struggles to adapt to the sudden loss of external guidance. In contrast, a gradual annealing allows the larger model to smoothly absorb the exploration benefits and progressively adapt its own policy to the high-quality regions discovered by the explorer, avoiding optimization divergence.

Ablation on transition length: insufficient annealing degrades performance. Finally, we analyze the impact of the transition duration. Fig. 7 compares schedules with different annealing lengths (e.g., transitioning over 8 steps vs. 5 steps). Results indicate that shortening the transition phase leads to worse training stability and a lower performance ceiling. This suggests that the handover from small-modelassisted rollouts to predominantly larger-model rollouts is a meaningful control knob: if the transition is too fast, the larger model may not have sufficient time to digest the diverse exploration signals before being forced back to its own limited distribution. Therefore, a sufficiently long annealing period is essential for maximizing the downstream gains of the proposed method.

- 5. Related Work

[Figure 94]

[Figure 95]

n = 12

n = 12

N=16N=8

[Figure 96]

[Figure 97]

n = 12 n = 12

- Figure 5. Pure small-model rollouts are insufficient for sustained improvement. Here N denotes the number of GRPO rollouts and n denotes the number of small-model rollouts, allowing to match total compute across settings.

0 2k 4k 6k 8k 10k 12k 14k 16k

Training data size

0.10

0.12

0.14

0.16

0.18

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Qwen3-8B-Base

Progressive transition

Abrupt switching

GRPO

- Figure 6. Progressive transition vs. abrupt two-phase switching.

cally requires expensive on-policy rollouts and maintaining multiple synchronized components (e.g., policy, reference model, and often a critic), leading to considerable engineering complexity and computational overhead. To simplify training, Direct Preference Optimization (DPO) (Rafailov et al., 2023) rewrites KL-regularized preference learning into a closed-form classification objective, avoiding online rollouts and an explicit critic, and thus substantially streamlining the pipeline. More recently, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) replaces the critic with group-relative advantage estimation using within-group statistics, reducing training cost while retaining PPO-style update stability, and is a standard baseline for reasoningoriented RL post-training. Nevertheless, RL post-training for reasoning can still be dominated by rollout cost and limited sample efficiency (Gao et al., 2025; Hassani et al., 2025; Lanchantin et al., 2025; Mroueh et al., 2025; Wang et al., 2025b; Yu et al., 2025; Zhang et al., 2025a; Zheng et al., 2025), especially for long-horizon reasoning tasks that require repeated sampling, scoring, and backpropagation over long sequences.

##### 5.1. The Evolution of RLVR

Reinforcement learning is a key paradigm for aligning large language models (LLMs) and improving reasoning ability in post-training, and recent practice is gradually shifting from preference-centric RLHF to RL with verifiable rewards (RLVR) that leverages automatically checkable signals (Guo et al., 2025; Kaufmann et al., 2023; Zhao et al., 2025). Early RLHF systems commonly relied on PPO (Schulman et al., 2017) as an online optimizer, but PPO-style training typi-

Qwen3-8B-Base

S2L-PO 8-step transition S2L-PO 5-step transition GRPO

0.18

0.16

| |
|---|

0.14

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.12

| |
|---|

| |
|---|

| |
|---|

0.10

0.08

0 2k 4k 6k 8k 10k 12k 14k 16k

Training data size

Figure 7. Ablation on transition length. We compare progressive annealing schedules that reduce the small-model rollout ratio to zero over the first 8 steps versus the first 5 steps.

##### 5.2. Diversity and Exploration in GRPO-Style Training

- A central practical factor for GRPO-style methods is the diversity of candidate trajectories sampled for each prompt: when the sampled group becomes overly homogeneous or degenerates, advantage estimation and gradient signals can deteriorate, potentially causing entropy collapse, mode collapse, and insufficient exploration (Hao et al., 2025; Jin et al., 2025; Yu et al., 2025). Most existing approaches encourage exploration by injecting randomness at the token level, e.g., via higher temperature, top-p sampling, or entropy regularization (Huang et al., 2025; Lin et al., 2024; Nguyen et al., 2024; Shi et al., 2024a; Wang et al., 2025c; Yang et al., 2025b;c; Zhuang et al., 2025). However, such action-space stochasticity is local and step-wise, and may not reliably yield trajectory-level structured diversity; moreover, aggressively increasing token uncertainty can hurt solution quality and training stability (Wang et al., 2025a). Beyond decoding-time randomness, several works improve group diversity through data- or objective-level interventions, such as selecting more diverse response types or explicitly rewarding within-group diversity (Anschel et al., 2025; Bamba et al., 2025; Chen et al., 2025; Zhang & Zuo, 2025; Zhang et al., 2025c). While effective in some settings, these approaches often require additional engineering or computation, and their gains may be less robust when transferring to new tasks or distributions.

Compared with token-level randomness and dataset-level heuristics, exploration via policy-level perturbation has received relatively less attention in LLM RL post-training. Recent off-policy methods (Chen et al., 2026; Lanchantin et al., 2025; Wang et al., 2025b) reuse previously generated rollouts or leverage external data to reduce sampling cost, but purely offline rollouts struggle to sustain performance improvement as the learner’s policy evolves, due to widening distribution shift. Ensemble-based approaches can provide diverse policies but require maintaining multiple models of comparable capacity, incurring significant additional cost. S2L-PO instead introduces policy-level diversity at near-zero cost by reusing an existing smaller model from the same family, and its progressive annealing

strategy ensures sustained improvement by smoothly transitioning from small-model exploration to on-policy learning, avoiding the performance plateau inherent in purely offline approaches.

### 6. Conclusion

We have presented S2L-PO, a new framework that enhances GRPO by utilizing smaller models as structured explorers for larger learners. Because smaller models obtained via parameter-level compression (e.g., distillation) inherently exhibit policy-level diversity, we provide empirical and theoretical evidence that adding this diversity to standard GRPO leads to more coherent exploration and improved learning signals than injecting token-level randomness alone. With a designed annealing strategy to balance exploration and exploitation, S2L-PO achieves significant gains in mathematical reasoning tasks while reducing rollout compute and accelerating convergence. Our results demonstrate that leveraging the inherent diversity from parameter-level perturbation is a powerful and efficient strategy for RL training.

### Impact Statement

This work exclusively relies on publicly available opensource datasets that have been widely used and validated in prior academic research. No new text, images, audio, or video content is generated or collected as part of this study. All datasets are used strictly for research purposes, and we do not engage in any commercial deployment or application of the data or the trained models.

### Acknowledgements

This work was partly supported by the National Natural Science Foundation of China (Grant No. 62576191), the Shenzhen Science and Technology Program (ZDCY20250901103533010) and Tsinghua SIGS KA Cooperation Fund.

### References

Anschel, O., Shoshan, A., Botach, A., Hakimi, S. H., Gendler, A., Baruch, E. B., Bhonker, N., Kviatkovsky, I., Aggarwal, M., and Medioni, G. Group-aware reinforcement learning for output diversity in large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 32382–32403, 2025.

Balunovi´c, M., Dekoninck, J., Petrov, I., Jovanovi´c, N., and Vechev, M. Matharena: Evaluating llms on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.

Bamba, U., Fang, M., Yu, Y., Zheng, H., and Lai, F. Xrpo: Pushing the limits of grpo with targeted exploration and exploitation. arXiv preprint arXiv:2510.06672, 2025.

Bansal, H., Hosseini, A., Agarwal, R., Tran, V. Q., and Kazemi, M. Smaller, weaker, yet better: Training llm reasoners via compute-optimal sampling. arXiv preprint arXiv:2408.16737, 2024.

Cai, Z., Cao, M., Chen, H., Chen, K., Chen, K., Chen, X., Chen, X., et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Chen, X., Zhu, W., Qiu, P., Dong, X., Wang, H., Wu, H., Li, H., Sotiras, A., Wang, Y., and Razi, A. Dra-grpo: Exploring diversity-aware reward adjustment for r1-zerolike training of large language models. arXiv preprint arXiv:2505.09655, 2025.

Chen, Z., Liu, H., Zhou, Y., Zheng, H., and Chen, B. Jackpot: Optimal budgeted rejection sampling for extreme actor-policy mismatch reinforcement learning, 2026. URL https://arxiv.org/abs/2602.06107.

Dragoi, M., Pintilie, I., Gogianu, F., and Brad, F. Beyond pass@ k: Breadth-depth metrics for reasoning boundaries. arXiv preprint arXiv:2510.08325, 2025.

Gao, C., Zheng, C., Chen, X.-H., Dang, K., Liu, S., Yu, B., Yang, A., Bai, S., Zhou, J., and Lin, J. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347, 2025.

- Gu, Y., Dong, L., Wei, F., and Huang, M. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024.
- Gu, Z., Chen, X., Shi, X., Wang, T., Zheng, S., Li, T., Feng, H., and Xiao, Y. Gapo: Learning preferential prompt through generative adversarial policy optimization. arXiv preprint arXiv:2503.20194, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hao, Z., Wang, H., Liu, H., Luo, J., Yu, J., Dong, H., Lin, Q., Wang, C., and Chen, J. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150, 2025.

Hassani, H., Hallaji, E., Razavi-Far, R., Saif, M., and Lin, L. Towards sample-efficiency and generalization of transfer and inverse reinforcement learning: A comprehensive literature review. IEEE Transactions on Artificial Intelligence, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Hong, J., Lee, N., and Thorne, J. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2024.

Huang, W., Ge, Y., Yang, S., Xiao, Y., Mao, H., Lin, Y., Ye, H., Liu, S., Cheung, K. C., Yin, H., et al. Qerl: Beyond efficiency–quantization-enhanced reinforcement learning for llms. arXiv preprint arXiv:2510.11696, 2025.

Jin, R., Gao, P., Ren, Y., Han, Z., Zhang, T., Huang, W., Liu, W., Luan, J., and Xiong, D. Revisiting entropy in reinforcement learning for large reasoning models. arXiv preprint arXiv:2511.05993, 2025.

Kaufmann, T., Weng, P., Bengs, V., and H¨ullermeier, E. A survey of reinforcement learning from human feedback. arXiv preprint arXiv:2312.14925, 2023.

Lanchantin, J., Chen, A., Lan, J., Li, X., Saha, S., Wang, T., Xu, J., Yu, P., Yuan, W., Weston, J. E., et al. Bridging offline and online reinforcement learning for llms. arXiv preprint arXiv:2506.21495, 2025.

Lin, Z., Liang, T., Xu, J., Lin, Q., Wang, X., Luo, R., Shi, C., Li, S., Yang, Y., and Tu, Z. Critical tokens matter: Token-level contrastive estimation enhances llm’s reasoning capability. arXiv preprint arXiv:2411.19943, 2024.

Mroueh, Y., Dupuis, N., Belgodere, B., Nitsure, A., Rigotti, M., Greenewald, K., Navratil, J., Ross, J., and Rios, J. Revisiting group relative policy optimization: Insights into on-policy and off-policy training. arXiv preprint arXiv:2505.22257, 2025.

Nguyen, M. N., Baker, A., Neo, C., Roush, A., Kirsch, A., and Shwartz-Ziv, R. Turning up the heat: Min-p sampling for creative and coherent llm outputs. arXiv preprint arXiv:2407.01082, 2024.

Park, Y.-J. and Cho, H.-S. Subset-aware dual-teacher knowledge distillation with hybrid scoring for human activity recognition. Electronics, 14(20):4130, 2025.

Peng, T. and Zhang, J. Enhancing knowledge distillation of large language models through efficient multi-modal distribution alignment. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 2478–2496, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shi, C., Yang, C., Zhu, X., Wang, J., Wu, T., Li, S., Cai, D., Yang, Y., and Meng, Y. Unchosen experts can contribute too: Unleashing moe models’ power by self-contrast. Advances in Neural Information Processing Systems, 37: 136897–136921, 2024a.

Shi, C., Yang, H., Cai, D., Zhang, Z., Wang, Y., Yang, Y., and Lam, W. A thorough examination of decoding methods in the era of llms. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8601–8629, 2024b.

Talmor, A., Herzig, J., Lourie, N., and Berant, J. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.

Wang, C., Li, Z., Bai, J., Zhang, Y., Cui, S., Zhao, Z., and Wang, Y. Arbitrary entropy policy optimization breaks the exploration bottleneck of reinforcement learning. arXiv preprint arXiv:2510.08141, 2025a.

Wang, H., Hao, S., Dong, H., Zhang, S., Bao, Y., Yang, Z., and Wu, Y. Offline reinforcement learning for llm multi-step reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 8881–8893, 2025b.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025c.

Wang, Y., Yang, Q., Zeng, Z., Ren, L., Liu, L., Peng, B., Cheng, H., He, X., Wang, K., Gao, J., et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025d.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, C., Gui, L., Yang, C., Veitch, V., Zhang, L., and Zhao, Z. Let it calm: Exploratory annealed decoding for verifiable reinforcement learning. arXiv preprint arXiv:2510.05251, 2025b.

Yang, C., Shi, C., Li, S., Shui, B., Yang, Y., and Lam, W. Llm2: Let large language models harness system 2 reasoning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 168–177, 2025c.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Zhang, H., Zheng, R., Yi, Z., Peng, H., Wang, H., and Yu, Y. Group expectation policy optimization for stable heterogeneous reinforcement learning in llms. arXiv eprints, pp. arXiv–2508, 2025a.

Zhang, J. and Zuo, C. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. arXiv preprint arXiv:2504.09696, 2025.

Zhang, X., Wen, S., Wu, W., and Huang, L. Edge-grpo: Entropy-driven grpo with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025b.

Zhang, X., Wu, S., Zhu, Y., Tan, H., Yu, S., He, Z., and Jia, J. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning. arXiv preprint arXiv:2510.19807, 2025c.

Zhao, Y., Liu, Y., Liu, J., Chen, J., Wu, X., Hao, Y., Lv, T., Huang, S., Cui, L., Ye, Q., et al. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zhuang, H., Zhou, Y., Guo, T., Huang, Y., Liu, F., Song, K., and Zhang, X. Exploring multi-temperature strategies for token-and rollout-level control in rlvr. arXiv preprint arXiv:2510.08892, 2025.

### A. Reproducibility Statement

To facilitate reproducibility and transparency, we will release the complete codebase of this project as open-source software. The overall methodology and algorithmic design are described in detail in Section 3. The experimental setup is specified in Section 4.1, and the complete training protocols, implementation details, and key hyperparameter configurations are provided in the appendix. These materials together are sufficient to reproduce all experimental results reported in this paper.

### B. Use of Large Language Models

During the preparation of this manuscript, we used a large language model solely for language editing purposes, including improving grammar, clarity, and overall readability at the sentence and paragraph levels. The model was not used to generate research ideas, design methods, conduct experiments, analyze results, or draw scientific conclusions. All technical content, experimental design, analyses, and interpretations were written, verified, and approved by the authors. Every model-assisted edit was carefully reviewed to ensure correctness, and the authors take full responsibility for the accuracy and integrity of the final manuscript.

### C. Hyperparameter Settings

All experiments are trained with GRPO using a fixed set of core hyperparameters across runs. We set the training batch size to 1024, the maximum prompt length to 512 tokens, and the maximum response length to 4096 tokens, while filtering overlong prompts and treating truncation as an error to avoid silent data corruption. The actor policy is optimized with a learning rate of 1 × 10−6 using PPO-style updates with a mini-batch size of 16 and a per-GPU micro-batch size of 2. We enable KL regularization between the actor and a reference policy with KL coefficient 1 × 10−3 and use a low-variance KL estimator, while setting the entropy bonus coefficient to 0 and not incorporating KL into the reward. For rollout and log-probability computation, we use micro-batching with size 2 per GPU, and keep tensor model parallelism at 1. All runs are performed on 8 GPUs, with checkpointing enabled at every training step and evaluation triggered every 100 steps, and training is terminated by a fixed number of training steps rather than a fixed number of epochs. We use a progressive off-policy to on-policy schedule over 16 logical steps, where the first 8 logical steps linearly decrease the offline sampling ratio from 1 to 0 and increase the online rollout ratio from 0 to 1, and the remaining logical steps are fully on-policy.

### D. Deployment of Progressive Off-to-On GRPO

In this appendix, we describe how our progressive off-policy to on-policy schedule is deployed within the GRPO training pipeline. The key idea is to generate candidate trajectories from a mixture of an offline source and the current policy, and to linearly anneal the offline contribution to zero. This staged procedure provides low-cost exploration early in training while ensuring that the final policy is optimized under its own on-policy distribution.

### E. Formal Proofs for Theoretical Analysis

In this appendix we provide rigorous proofs for the theoretical claims in Section 3.1. We formalize the distinction between token-level and policy-level perturbations in terms of cross-step covariance bounds for the GRPO gradient signal.

##### E.1. Notation and Setup

Let o = (a1,...,aL) denote a sampled trajectory for prompt q, and o⋆ = (a⋆1,...,a⋆L) a deterministic reference decoding trace under the base policy πθ. Define the prefix match indicator Mt := I{(a1,...,at) = (a⋆1,...,a⋆t)}. For the per-step score function ui,t := ∇θ log πθ(ai,t | si,t), define the scalar projection zi,t := ⟨ui,t,v⟩ for a unit vector v, and assume |zi,t| ≤ B.

Lemma E.1 (Prefix Match Decay). Under token-level perturbation with per-step deviation probability lower-bounded by p > 0, the prefix match probability satisfies:

Pr(Mt = 1) =

t

Pr(aj = a⋆j | Mj−1 = 1) ≤ (1 − p)t.

j=1

Proof. Since {Mt = 1} = {Mt−1 = 1}∩ {at = a⋆t}, the chain rule gives Pr(Mt = 1) = tj=1 Pr(aj = a⋆j | Mj−1 = 1), with base case Pr(M0 = 1) = 1. The assumption Pr(aj ̸= a⋆j | Mj−1 = 1) ≥ p yields each factor ≤ 1 − p, so the product ≤ (1 − p)t.

| |
|---|

Proposition E.2 (Token-Level Covariance Upper Bound). Let ft,gs be random variables measurable with respect to the trajectory prefix (a1,...,at) and (a1,...,as) respectively, with ∥ft∥∞,∥gs∥∞ ≤ 1 and t < s. Then:

###### |Cov(ft,gs | q)| ≤ 5 Pr(Mt = 0).

Proof. We apply the law of total covariance with conditioning variable Mt ∈ {0,1}:

+Cov(E[ft | Mt],E[gs | Mt])

Cov(ft,gs) = E[Cov(ft,gs | Mt)]

.

(I)

(II)

- Term (I): When Mt = 1, the prefix (a1,...,at) is fixed to (a⋆1,...,a⋆t), so ft is a constant and Cov(ft,gs | Mt = 1) = 0. When Mt = 0, Cauchy–Schwarz gives |Cov(ft,gs | Mt = 0)| ≤ 1. Thus |(I)| ≤ Pr(Mt = 0) · 1.
- Term (II): Let α = Pr(Mt = 1), µk = E[ft | Mt = k], νk = E[gs | Mt = k] for k ∈ {0,1}. Since E[ft | Mt] and E[gs | Mt] are functions of a Bernoulli variable:

(II) = α(1 − α)(µ1 − µ0)(ν1 − ν0). Using α(1 − α) ≤ 1 − α = Pr(Mt = 0) and |µ1 − µ0|,|ν1 − ν0| ≤ 2:

|(II)| ≤ 4 Pr(Mt = 0).

Combining: |Cov(ft,gs)| ≤ Pr(Mt = 0) + 4 Pr(Mt = 0) = 5 Pr(Mt = 0). Corollary E.3 (Gradient Projection Covariance). For the scalar projections zi,t = ⟨ui,t,v⟩ with |zi,t| ≤ B and t < s:

###### |Cov(zi,t,zi,s | q)| ≤ 5B2 Pr(Mt = 0).

| |
|---|

Proof. Apply Proposition E.2 to ft = zi,t/B and gs = zi,s/B, then scale by B2.

| |
|---|

Proposition E.4 (Policy-Level Covariance Lower Bound). Let z˜i,t = zi,t + v⊤Htδθ denote the first-order perturbed score projection, where Ht = ∇2θ log πθ(at | st) and δθ has zero mean and covariance Σδ. If γ := E[v⊤HtΣδHs⊤v | q] > 0, then for t < s:

std Pr(Mt = 0) − O(∥δθ∥3),

|Cov(˜zi,t,z˜i,s | q)| ≥ γ − 5B2

where Prstd denotes prefix divergence under the standard (unperturbed) temperature.

Proof. Expanding by bilinearity of covariance:

###### Cov(˜zt,z˜s) = Cov(zt,zs) + Cov(zt,v⊤Hsδθ) + Cov(v⊤Htδθ,zs) + Cov(v⊤Htδθ,v⊤Hsδθ).

Cross-terms vanish: In the zero-order approximation, zt and Hs are computed along the base policy trajectory and are independent of δθ. Since E[δθ] = 0, both E[zt ·v⊤Hsδθ] and E[zt]·E[v⊤Hsδθ] vanish, giving Cov(zt,v⊤Hsδθ) = 0. The trajectory’s O(∥δθ∥) dependence on δθ contributes O(∥δθ∥3) to the covariance.

Perturbation term: Both means vanish (E[δθ] = 0), so:

Cov(v⊤Htδθ,v⊤Hsδθ) = E[(v⊤Htδθ)(δθ⊤Hs⊤v)] = E[v⊤HtΣδHs⊤v | q] = γ, where we used the independence of δθ and the trajectory (at zero order) to factor the expectation. Combining: Cov(˜zt,z˜s) = Cov(zt,zs) + γ + O(∥δθ∥3). By the reverse triangle inequality and Corollary E.3:

std Pr(Mt = 0) − O(∥δθ∥3).

|Cov(˜zt,z˜s)| ≥ γ − |Cov(zt,zs)| − O(∥δθ∥3) ≥ γ − 5B2

The lower bound is positive when γ dominates, i.e., when Hessian alignment is strong and the standard-temperature prefix divergence is moderate.

| |
|---|

- Remark E.5. The Hessian alignment condition γ > 0 is natural for same-family distilled models: since δθ arises from structured capacity reduction that affects shared feature-extraction layers, the Hessians Ht and Hs project δθ onto similar directions across different decoding steps, yielding consistently positive alignment. This contrasts sharply with token-level

perturbation, where the covariance upper bound in Corollary E.3 becomes vacuous as Pr(Mt = 0) → 1 with increasing temperature.

- Remark E.6 (Fixed vs. random δθ). In practice, δθ = θsmall − θlarge is a fixed vector determined by the specific small model. In the theoretical analysis above, we model δθ as a zero-mean random variable (representing the uncertainty across possible distillation outcomes) in order to derive the expectation-based lower bound. For a fixed δθ, the covariance expansion still holds: the fourth term becomes Cov(v⊤Htδθ,v⊤Hsδθ | q), which is the covariance of Hessian projections over trajectory randomness. The lower bound then depends on the specific alignment of this fixed δθ with the Hessian structure, rather than on the averaged alignment γ.
- Remark E.7 (Bound on B under softmax parameterization). Under softmax parameterization, log πθ(a | s) = ℓa −

log a′ exp(ℓa′), where ℓa denotes the logit for action a. The gradient with respect to the logit parameters is ∇ℓ log πθ(a | s) = ea − πθ(· | s), where ea is a one-hot vector. Each component has absolute value at most 1, so ∥∇ℓ log πθ(a | s)∥2 ≤

|V| where |V| is the vocabulary size. Consequently, for the scalar projection zi,t = ⟨ui,t,v⟩ with ∥v∥ = 1, one can take

- B = |V| in the logit-parameter setting. For the full model parameters θ (beyond the last layer), gradient clipping during training provides an effective bound on B.

- E.2. Summary: Token-Level vs. Policy-Level Signal Growth The key qualitative difference between the two perturbation mechanisms can be summarized as follows:

Token-level (high temp.) Policy-level (param. perturb.)

Covariance Upper bound: ≤ 5B2 Prtok(Mt=0) Lower bound: ≥ γ − 5B2 Prstd(Mt=0) − O(∥δθ∥3) Behavior Prtok(Mt=0) → 1 fast ⇒ vacuous γ > 0 independent of t ⇒ positive Signal growth E[∥ t ut∥2] ∼ O(L) (random walk) E[∥ t u˜t∥2] ≳ O(L2) (constructive)

Policy-level perturbation injects a shared, time-invariant signal v⊤Htδθ into each step’s score function. This common component induces positive cross-step covariance, causing gradient contributions to reinforce constructively across the horizon rather than cancelling like a random walk. This is the theoretical basis for why smaller models, as structured policy perturbations, provide more informative exploration signals for GRPO training.

### F. Limitations

Our empirical evaluation is constrained by computational resources, preventing exhaustive coverage of all prominent model families and benchmark categories. In particular, we have not validated S2L-PO on tasks beyond mathematical reasoning that rely on non-verifiable or open-ended rewards. The capability boundary of S2L-PO under broader model scales, task domains, and modalities remains to be explored in future work.

