[Figure 1]

2026-09-02

## iGRPO: Self-Feedback–Driven LLM Reasoning

Ali Hatamizadeh1, Shrimai Prabhumoye, Igor Gitman, Ximing Lu, Seungju Han, Wei Ping, Yejin Choi, Jan Kautz

# arXiv:2602.09000v1[cs.AI]9Feb2026

### Abstract

Large Language Models (LLMs) have shown promise in solving complex mathematical problems, yet they still fall short of producing accurate and consistent solutions. Reinforcement Learning (RL) is a framework for aligning these models with task-specific rewards, improving overall quality and reliability. Group Relative Policy Optimization (GRPO) is an efficient, value-function-free alternative to Proximal Policy Optimization (PPO) that leverages group-relative reward normalization. We introduce Iterative Group Relative Policy Optimization (iGRPO), a two-stage extension of GRPO that adds dynamic selfconditioning through model-generated drafts. In Stage 1, iGRPO samples multiple exploratory drafts and selects the highest-reward draft using the same scalar reward signal used for optimization. In Stage 2, it appends this best draft to the original prompt and applies a GRPO-style update on draft-conditioned refinements, training the policy to improve beyond its strongest prior attempt. Under matched rollout budgets, iGRPO consistently outperforms GRPO across base models (e.g., Nemotron-H-8B-Base-8K and DeepSeek-R1 Distilled), validating its effectiveness on diverse reasoning benchmarks. Moreover, applying iGRPO to OpenReasoning-Nemotron-7B trained on AceReason-Math achieves new state-of-theart results of 85.62% and 79.64% on AIME24 and AIME25, respectively. Ablations further show that the refinement wrapper generalizes beyond GRPO variants, benefits from a generative judge, and alters learning dynamics by delaying entropy collapse. These results underscore the potential of iterative, self-feedback-based RL for advancing verifiable mathematical reasoning.

### 1. Introduction

Reinforcement Learning (RL) has proven to be successful in improving reasoning capabilities of LLMs by optimizing against task-specific reward signals. Early successes in this direction include RL from Human Feedback (RLHF) for aligning LLMs with human intent, most notably in InstructGPT (Ouyang et al., 2022) and ChatGPT (Achiam et al., 2023), which have demonstrated that incorporating preference-based rewards can dramatically improve both the usability and correctness of model outputs. Recently DeepSeek-R1 (Guo et al., 2025) proposed a distinguishing feature which is the so-called zero configuration, wherein the RL process directly enhances the base language model. This breakthrough started several efforts which were targeted at replicating DeepSeek-R1’s methodology or refining its underlying RL mechanisms (Zeng et al., 2025; Yu et al., 2025; Liu et al., 2025; Cui et al., 2025; Hu et al., 2025).

Yet, in the realm of complex reasoning, RL algorithms typically do not incorporate any form of feedback or reflection on the model’s own outputs. Humans, by contrast, rarely solve nontrivial problems in a single pass: they often iterate on initial drafts, identify mistakes, and refine their solutions based on internal feedback (Flower & Hayes, 1981; Simon, 2012; Braidotti, 2019; Flavell, 1979; Schön, 2017; Polya, 2014). There is growing evidence that self-feedback mechanisms can bolster multi-step reasoning and the capacity to correct errors (Madaan et al., 2023; Shinn et al., 2023). However, existing RL frameworks do not capitalize on this iterative refinement process, leaving a critical gap between how humans naturally solve problems and how LLMs are typically trained to do so.

In this work, we propose to fill this gap with Iterative GRPO (iGRPO) which is a powerful extension of GRPO (Shao et al., 2024). As illustrated in Figure 1, Our method operates in two stages. First, we draw multiple candidate completions from the model and compute their relative rewards via the group-based mechanism

1Project lead. Correspondence to: Ali Hatamizadeh<ahatamizadeh@nvidia.com>.

© 2026 NVIDIA. All rights reserved.

[Figure 2]

iGRPO: Self-Feedback–Driven LLM Reasoning

- Figure 1: Iterative GRPO (iGRPO): During Exploratory Draft Generation, the model selects a high-scoring “best draft” from initial samples and appends it to the prompt for Conditioned Refinement. This augmented context guides the generation of new group-based updates, creating a bootstrapping effect where the policy continuously improves its own conditioning signal to enhance reasoning.

of GRPO. We then select the highest-scoring draft and this serves as the "first-draft" output of the model. We consider this highest-scoring response as a guide to improve the final output. Hence, it is provided as a self-feedback to the model. We feed it back to the model alongside the original prompt. By conditioning on this exemplar, the second stage encourages the model to refine and surpass its own best prior attempt. Notably, this design preserves the efficiency of GRPO while introducing only minimal extra overhead, as iGRPO still relies on the same set of group-based reward signals. In doing so, iGRPO offers a promising avenue for self-guided improvement, enabling LLMs to iteratively improve their reasoning capabilities.

We conduct a series of controlled experiments to compare iGRPO and GRPO under identical training conditions, using different base models trained on the Mathematics Aptitude Test of Heuristics (MATH) (Hendrycks et al., 2021) dataset. Specifically, we evaluate DeepSeek-R1 Distilled (Guo et al., 2025) and OpenMathNemotron (Moshkov et al., 2025) on an extensive array of mathematical reasoning benchmarks, including AIME24 (AI-MO, 2024a), AIME25 (OpenCompass, 2025), MATH500 (Lightman et al., 2023), AMC23 (AI-MO,

- 2024b), GSM8K (Cobbe et al., 2021), and Minerva Math (Lewkowycz et al., 2022). For models with 7B and 14B parameters, iGRPO consistently outperforms standard GRPO. Moreover, by leveraging iGRPO algorithm with OpenReasoning-Nemotron-7B model (NVIDIA, 2025) on the large-scale AceReason-Math (Chen et al.,
- 2025b) dataset (R1, 2024), we push the state of the art on AIME24 and AIME25 to 85.62% and 79.64%, respectively. These findings underscore the effectiveness of incorporating a self-feedback stage into group-based RL optimization, particularly for complex mathematical reasoning tasks.

### 2. Related Work

##### RL for Reasoning.

Reinforcement learning (RL) has become an important tool for refining large language models on logical and analytical tasks (Lambert et al., 2024). Early self-improvement lines of work, such as STaR-style bootstrapping with verified outcomes and sampling-based selection (as discussed in (Lambert et al., 2024)), demonstrate that iteratively leveraging model-generated solutions can improve reasoning behavior. More recent systems scale outcome-driven training substantially (Jaech et al., 2024), and open-weight efforts such as DeepSeek-R1 report strong reasoning performance under similar training regimes (Guo et al., 2025). Beyond natural language tasks, RL on procedurally generated puzzles (Xie et al., 2025) and settings with limited human demonstrations

(Wang et al., 2025) further highlight the breadth of RL as a mechanism for improving mathematical and logical reasoning.

##### GRPO and Variants.

There has also been rapid progress in refining and extending GRPO for large-scale LLM training. Dr. GRPO (Liu et al., 2025) analyzes sources of bias in GRPO-style token-level objectives and proposes modifications such as removing divisions by sequence length and group-level standard deviation to better match unbiased policy gradients. DAPO (Yu et al., 2025) targets long chain-of-thought training through dynamic sampling, decoupled clipping, and reward shaping designed to mitigate instability and reward noise. GSPO (Zheng et al., 2025) instead operates at the sequence level, redefining importance ratios and applying sequence-level clipping to improve stability, especially in Mixture-of-Experts settings. Whereas these approaches primarily focus on stabilizing or correcting the underlying optimization objective, iGRPO is orthogonal: it introduces a two-stage mechanism that uses externally evaluated best drafts as additional training context, shaping the data distribution seen by the optimizer.

##### LLM Self-Learning and Self-Improvement.

LLM self-learning methods aim to improve a model by leveraging feedback signals produced by the model itself or by closely related agents. For example, SPIN and Self-Rewarding Language Models (Chen et al., 2024; Yuan et al., 2024) use the model (or a closely coupled variant) as an internal evaluator to drive further learning. Other approaches incorporate self-play, verifier-based alignment, or proof-oriented training signals (Kirchner et al., 2024; Ye et al., 2024), though unreliable rewards can hinder complex reasoning (Lambert et al., 2024). More specialized self-play extensions, such as SPC (Chen et al., 2025a) and SPAG (Cheng et al., 2024), introduce curated tasks or adversarial scenarios to strengthen critique and robustness. Self-Verification (Zhang et al., 2025a) unifies problem-solving and generative verification within a single RL framework, enabling inference-time scaling by using the model’s own verification scores to reweight or aggregate sampled solutions. Critique-GRPO (Zhang et al., 2025b) augments GRPO with natural-language critiques by generating critiqueconditioned refinements and optimizing over both initial answers and their refinements in an online RL loop. While these methods often blur the roles of reward provider and reward recipient within a single agent (or paired agents), iGRPO instead uses externally evaluated best-prior drafts as an in-context guide for subsequent generations during training. This maintains a clearer separation between the model’s generation process and the reward signal, while still leveraging the core self-improvement principle of learning from the model’s own outputs.

### 3. Methodology

#### 3.1. Background: Group Relative Policy Optimization

GRPO is a value-function-free variant of proximal policy optimization that leverages group-based relative rewards for advantage estimation. Let 𝜋𝜃 denote the current policy and 𝜋𝜃

the old policy. We begin with a pretrained language model ℳ and a set of training instances ℬ = {(𝑞,𝑎)}, where 𝑞 is a prompt (e.g., a math problem) and 𝑎 is a reference answer. Given a prompt 𝑞, GRPO samples a group of 𝐺 candidate outputs {𝑜1,𝑜2,...,𝑜𝐺} from 𝜋𝜃

old

:

old

(· | 𝑞) for 𝑖 = 1,...,𝐺.

𝑜𝑖 ∼ 𝜋𝜃

old

A reward model then evaluates each sampled output 𝑜𝑖, resulting in scores {𝑅1,...,𝑅𝐺}. GRPO normalizes these scores within the group to compute the advantage 𝐴ˆ𝑖,𝑡 at each token index 𝑡 of 𝑜𝑖:

𝑅𝑖 − mean({𝑅1,...,𝑅𝐺}) std({𝑅1,...,𝑅𝐺})

𝐴ˆ𝑖,𝑡 =

,

where mean(·) and std(·) denote the sample mean and sample standard deviation of the group’s reward scores. If std({𝑅1,...,𝑅𝐺}) = 0, we set the normalized advantages to 0 (equivalently, one may add a small constant 𝛿 to the denominator); we apply the same convention in iGRPO. Note that all tokens 𝑡 in 𝑜𝑖 share the same advantage 𝐴ˆ𝑖,𝑡 = 𝐴ˆ𝑖, reflecting a single scalar reward for each sampled completion.

GRPO then updates the current policy 𝜋𝜃 by maximizing a clipped surrogate objective. Let

𝑟𝑖,𝑡(𝜃) =

𝜋𝜃(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃

.

(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡)

old

Then the GRPO objective is:

𝒥GRPO(𝜃) = E[︁𝑞 ∼ 𝑃(𝑄), {𝑜𝑖}𝐺𝑖=1 ∼ 𝜋𝜃old]︁

[︁min(︁𝑟𝑖,𝑡(𝜃) 𝐴^𝑖,𝑡, clip

𝐴^𝑖,𝑡)︁ − 𝛽 𝐷̂︀KL(𝑖,𝑡)]︁, (1)

∑︁|𝑜𝑖|

∑︁𝐺

(︀

)︀

1 |𝑜𝑖|

1 𝐺

𝑟𝑖,𝑡(𝜃), 1 − 𝜖, 1 + 𝜖

×

𝑡=1

𝑖=1

where 𝜖 is the PPO clipping parameter, 𝛽 is a regularization coefficient on the KL divergence to a reference policy 𝜋ref, and |𝑜𝑖| denotes the token length of completion 𝑜𝑖. We use the following non-negative per-token estimator 𝐷̂︀KL(𝑖,𝑡) as a practical KL penalty Schulman (2020):

𝜋ref(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) − log

𝜋ref(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃(𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) − 1,

𝐷̂︀KL(𝑖,𝑡) =

which remains guaranteed to be non-negative. This estimator is unbiased for 𝐷KL(𝜋𝜃 ‖𝜋ref) when the expectation is taken under samples from 𝜋𝜃, and it serves as a convenient sample-based penalty within PPO-style updates. By directly computing group-based advantages instead of estimating value functions, GRPO avoids the overhead of training a separate critic model, making it particularly appealing for large-scale language model fine-tuning.

#### 3.2. Iterative Group Relative Policy Optimization

We introduce Iterative Group Relative Policy Optimization (iGRPO), a reinforcement learning algorithm that implements bootstrapped policy improvement through dynamic self-conditioning. Unlike standard policy gradient methods that optimize single-shot generations, iGRPO establishes a closed-loop refinement process where the policy learns to systematically improve upon its own best attempts. The key insight is that by coupling exploration (Stage 1) with conditioned exploitation (Stage 2) within each optimization step, the model acquires a generalizable self-improvement capability that compounds in training.

##### 3.2.1. Motivation: From Static Examples to Dynamic Self-Conditioning

The standard GRPO objective (Eq. 1) treats each generation as independent, ignoring potentially valuable information from the model’s own generation process. In-context learning (ICL) addresses this by conditioning on fixed demonstrations:

[︀

]︀

𝒥ICL(𝜃) = E𝑞∼𝑃(𝑄)

E𝑜∼𝜋

𝜃(·|𝑞,𝑒) [𝑅𝜑(𝑜)]

,

where 𝑒 is a static example. However, ICL suffers from a fundamental limitation: the conditioning signal 𝑒 remains fixed throughout training and does not adapt to the evolving policy’s capabilities.

iGRPO introduces a different paradigm, dynamic self-conditioning, where the conditioning signal is generated by the policy itself and co-evolves with learning:

𝒥iGRPO(𝜃) = E𝑞∼𝑃(𝑄)[︁E𝑜∼𝜋

𝜃(·|𝑞𝜃′(𝑞))[𝑅𝜑(𝑜)]]︁, 𝑞𝜃′ (𝑞) = Concat

(︀

)︀

𝑞,𝑑ˆ𝜃(𝑞)

where 𝑑ˆ𝜃(𝑞) is the best draft generated by the current policy (defined formally below). This creates a bootstrapped learning dynamic: as 𝜋𝜃 improves, so does the quality of 𝑑ˆ𝜃, which in turn provides increasingly informative conditioning for subsequent generations.

Although the expressions above are written in terms of 𝜋𝜃 for clarity, iGRPO follows the PPO/GRPO convention for stable optimization. At each iteration, we take a snapshot 𝜋𝜃

, sample both Stage 1 drafts and Stage 2 completions from 𝜋𝜃

old

, and update 𝜃 using importance ratios computed relative to 𝜋𝜃

. Stage 1 is

old

old

not differentiated through, but the distribution of selected drafts 𝑑ˆ(and thus the self-conditioned prompts 𝑞′) shifts as 𝜃 evolves across iterations. In addition, dynamic self-conditioning is used only during training. At inference time, we use the trained policy in the standard single-shot manner, generating directly from the original prompt 𝑞 without any draft generation, conditioning, or specialized selection scheme.

##### 3.2.2. Algorithmic Framework

iGRPO operates through two tightly coupled stages within each optimization step. Crucially, only Stage 2 outputs receive gradient updates, while Stage 1 serves as an adaptive exploration mechanism that shapes the optimization landscape.

- Stage 1: Exploratory Draft Generation. Given a prompt 𝑞, we sample 𝑁 candidate drafts from the current policy snapshot:

𝑑𝑖 ∼ 𝜋𝜃

old

(· | 𝑞), 𝑖 = 1,...,𝑁.

Each draft is evaluated using a reward function 𝑅𝜑, and we identify the highest-scoring draft:

𝑑ˆ= arg max

𝑖∈{1,...,𝑁}

𝑅𝜑(𝑑𝑖). (2)

This stage performs implicit curriculum generation: early in training, 𝑑ˆmay be a weak solution, but as the policy improves, 𝑑ˆincreasingly represents a high-quality attempt that approaches (but does not yet reach) optimal performance.

- Stage 2: Conditioned Refinement. We form an augmented prompt by appending the best draft immediately after the original prompt:

𝑞′ = Concat(𝑞,𝑑ˆ). (3) We then sample a group of 𝐺 completions, mirroring the standard GRPO sampling:

(· | 𝑞′), 𝑗 = 1,...,𝐺.

𝑜𝑗 ∼ 𝜋𝜃

old

These completions are scored, and GRPO-style advantage estimation and policy updates are applied exclusively to these Stage 2 outputs. In practice, the concatenation in Eq. 3 uses a fixed prompt template, which we provide in the supplementary materials.

##### 3.2.3. Theoretical Analysis: Bootstrapped Policy Improvement Definition 1 (Self-Conditioned Prompt Construction). Given a prompt 𝑞 and policy 𝜋𝜃, sample 𝑁 drafts

𝑑𝑖 ∼ 𝜋𝜃(· | 𝑞), 𝑖 = 1,...,𝑁, and select the best draft

𝑑ˆ𝜃(𝑞) = arg max

𝑅𝜑(𝑑𝑖).

𝑖∈{1,...,𝑁}

We then define the self-conditioned prompt by appending the best draft immediately after the original prompt:

(︀

)︀

𝑞𝜃′ (𝑞) = Concat

𝑞,𝑑ˆ𝜃(𝑞)

.

Unlike static ICL where the conditioning is independent of 𝜃, iGRPO’s conditioning is policy-dependent:

the constructed prompt 𝑞𝜃′ (𝑞) changes as 𝜃 changes, since 𝑑ˆ𝜃(𝑞) is generated by 𝜋𝜃. This creates a coupled dynamical system.

Proposition 3.1 (Progressive Conditioning Quality for Binary Rewards). Assume the reward is binary, 𝑅𝜑(𝑜) ∈ {0,1}, and Stage 1 drafts {𝑑𝑖}𝑁𝑖=1 are sampled i.i.d. from 𝜋𝜃(·|𝑞). Let

###### 𝑉𝜃(𝑞) = E𝑜∼𝜋

𝜃(·|𝑞)[𝑅𝜑(𝑜)] denote the expected reward under policy 𝜋𝜃, which equals the success probability 𝑝𝜃(𝑞) = Pr[𝑅𝜑(𝑜) = 1] in the binary case. Then the expected reward of the selected best draft 𝑑ˆ𝜃(𝑞) = arg max𝑖 𝑅𝜑(𝑑𝑖) satisfies

[︀

]︀

𝑅𝜑(𝑑ˆ𝜃(𝑞))

= 1 − (1 − 𝑉𝜃(𝑞))𝑁,

E

which is monotonically increasing in 𝑉𝜃(𝑞). Consequently, if optimization increases 𝑉𝜃(𝑞), then E[𝑅𝜑(𝑑ˆ𝜃(𝑞))] also increases, improving the conditioning quality for subsequent iterations in expectation.

Proof. For binary rewards, 𝑅𝜑(𝑑ˆ𝜃(𝑞)) = 1 if and only if at least one of the 𝑁 sampled drafts achieves reward 1. Under i.i.d. sampling, this occurs with probability

[︀∀𝑖, 𝑅𝜑(𝑑𝑖) = 0

]︀

= 1 − (1 − 𝑉𝜃(𝑞))𝑁.

1 − Pr

Since 𝑅𝜑(𝑑ˆ𝜃(𝑞)) ∈ {0,1}, its expectation equals this probability. The function 1 − (1 − 𝑥)𝑁 is increasing in 𝑥 ∈ [0,1], establishing monotonicity in 𝑉𝜃(𝑞).

| |
|---|

This proposition captures the bootstrapping effect: better policies generate better drafts, which provide more informative conditioning, which enables learning better policies. The model does not merely learn to copy the conditioning; it learns a refinement function that maps draft attempts to improved solutions.

##### 3.2.4. Mathematical Formulation

Building on the GRPO objective in Eq. 1, we present the complete iGRPO formulation. Let 𝜋𝜃 denote the policy being optimized, with 𝜋𝜃

representing the policy snapshot at the start of each iteration for importance sampling.

old

- Stage 1: Draft Selection. For each prompt 𝑞, Stage 1 samples 𝑁 drafts and selects the best:

{𝑑1,...,𝑑𝑁} ∼ 𝜋𝜃

old

(· | 𝑞), 𝑑ˆ= arg max

𝑖∈{1,...,𝑁}

𝑅𝜑(𝑑𝑖).

- Stage 2: Conditioned Generation and Advantage Computation. We form the augmented prompt by appending the selected draft to 𝑞 (Eq. 3) and sample 𝐺 completions:

(· | 𝑞′).

###### {𝑜1,...,𝑜𝐺} ∼ 𝜋𝜃

old

Each completion 𝑜𝑗 receives a reward 𝑅𝜑(𝑜𝑗), and advantages are computed via group normalization as in standard GRPO:

𝑅𝜑(𝑜𝑗) − mean({𝑅𝜑(𝑜1),...,𝑅𝜑(𝑜𝐺)}) std({𝑅𝜑(𝑜1),...,𝑅𝜑(𝑜𝐺)})

. (4)

𝐴ˆ𝑗 =

If std({𝑅𝜑(𝑜1),...,𝑅𝜑(𝑜𝐺)}) = 0, we set the normalized advantages to 0 (equivalently, one may add a small constant 𝛿 to the denominator). All tokens 𝑡 in 𝑜𝑗 share the same advantage 𝐴ˆ𝑗,𝑡 = 𝐴ˆ𝑗, consistent with the GRPO formulation.

Full Objective. The complete iGRPO objective combines both stages: 𝒥iGRPO(𝜃) = E[︁𝑞 ∼ 𝑃(𝑄)]︁ E[︁{𝑑𝑖}𝑁𝑖=1 ∼ 𝜋𝜃

]︁

, 𝑑ˆ= arg max

𝑅𝜑(𝑑𝑖), 𝑞′ = Concat(𝑞,𝑑ˆ), {𝑜𝑗}𝐺𝑗=1 ∼ 𝜋𝜃

###### (· | 𝑞′) ⏟ ⏞

###### (· | 𝑞) ⏟ ⏞

old

old

𝑖

Stage 1

Stage 2

∑︁|𝑜𝑗|

𝐴ˆ𝑗)︁ − 𝛽 𝐷̂︀KL(𝑗,𝑡)]︁, (5)

[︁min(︁𝑟𝑗,𝑡(𝜃)𝐴ˆ𝑗, clip

∑︁𝐺

(︀

)︀

1 |𝑜𝑗|

1 𝐺

𝑟𝑗,𝑡(𝜃), 1 − 𝜖, 1 + 𝜖

×

𝑡=1

𝑗=1

Algorithm 1 Iterative Group Relative Policy Optimization (iGRPO) Require: Pretrained model ℳ, training set ℬ = {(𝑞,𝑎)}, reward function 𝑅𝜑, draft count 𝑁, group size 𝐺,

clipping parameter 𝜖, KL coefficient 𝛽, iterations 𝐼, batch size 𝑆

- 1: Initialize 𝜋𝜃 ← ℳ, 𝜋ref ← ℳ
- 2: for iteration = 1,...,𝐼 do
- 3: 𝜋𝜃

old ← 𝜋𝜃 ◁ Snapshot policy for sampling

- 4: Sample batch {(𝑞(𝑘),𝑎(𝑘))}𝑆𝑘=1 ∼ ℬ
- 5: for 𝑘 = 1,...,𝑆 do
- 6: // Stage 1: Draft Generation
- 7: Sample drafts {𝑑(𝑖𝑘)}𝑁𝑖=1 ∼ 𝜋𝜃

old

(· | 𝑞(𝑘))

- 8: Compute rewards {𝑅𝜑(𝑑(𝑖𝑘))}𝑁𝑖=1 using reference answer 𝑎(𝑘)
- 9: Select best draft: 𝑑ˆ(𝑘) ← arg max𝑖 𝑅𝜑(𝑑(𝑖𝑘))
- 10: // Stage 2: Conditioned Refinement
- 11: Construct augmented prompt: 𝑞′(𝑘) ← Concat(𝑞(𝑘),𝑑ˆ(𝑘))
- 12: Sample refinements {𝑜(𝑗𝑘)}𝐺𝑗=1 ∼ 𝜋𝜃

old

(· | 𝑞′(𝑘))

- 13: Compute rewards {𝑅𝜑(𝑜(𝑗𝑘))}𝐺𝑗=1
- 14: Compute advantages {𝐴ˆ(𝑗𝑘)} via Eq. 4
- 15: end for
- 16: // Policy Update
- 17: 𝜃 ← 𝜃 + 𝜂∇𝜃𝒥iGRPO(𝜃) ◁ See Eq. 5
- 18: end for
- 19: return 𝜋𝜃

where the importance sampling ratio is computed with respect to the augmented prompt:

𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃

𝑟𝑗,𝑡(𝜃) =

,

(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡)

old

and we use the same non-negative per-token KL penalty as in Section 3.1, computed under the augmented prompt:

𝜋ref(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) − log

𝜋ref(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) − 1.

𝐷̂︀KL(𝑗,𝑡) =

The key structural difference from vanilla GRPO (Eq. 1) is that all Stage 2 completions are conditioned on the augmented prompt 𝑞′, formed by appending the best Stage 1 draft immediately after the original prompt 𝑞. This provides a self-feedback signal that encourages the policy to refine its solutions beyond its strongest initial attempt.

Reward Function. Following standard practice for verifiable reasoning tasks, we employ a rule-based reward function:

𝑅𝜑(𝑜) = 1[extract(𝑜) = 𝑎], (6)

where extract(·) parses the final answer from the completion and 𝑎 is the ground-truth reference answer from the training instance (𝑞,𝑎) ∈ ℬ. This binary reward, combined with group normalization (Eq. 4), provides sufficient signal for distinguishing solution quality within each prompt group. The complete iGRPO procedure is presented in Algorithm 1.

##### 3.2.5. Computational Analysis

Although iGRPO uses two stages, its compute is primarily controlled by the total number of sampled completions per prompt. With a fixed sampling budget, iGRPO can be run at essentially the same dominant generation cost as GRPO.

Table 1: Performance comparison of 7B, 8B, and 14B models across multiple mathematical reasoning benchmarks. Bold indicates the best and underlined the second best per column within each parameter bucket (7B, 8B, or 14B). Our method iGRPO rows are lightly shaded.

Model AIME25 AIME24 MATH500 AMC GSM8K Minerva Avg Nemotron-H-8B-Base-8K 6.20 8.65 61.23 43.21 41.02 17.60 29.65 Nemotron-H-8B-Base-8K + GRPO 7.78 9.01 73.13 45.10 81.93 29.56 41.08 Nemotron-H-8B-Base-8K + Self-Verification (Zhang et al., 2025a) 8.50 9.25 75.60 46.50 86.20 31.10 42.86 Nemotron-H-8B-Base-8K + Critique-GRPO (Zhang et al., 2025b) 8.42 9.15 76.05 46.80 88.40 31.50 43.39 Nemotron-H-8B-Base-8K + iGRPO 9.17 9.56 78.80 48.75 91.26 32.72 45.04 DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025) 38.60 54.40 92.80 90.00 92.00 39.10 61.93 DeepSeek-R1-Distill-Qwen-7B + GRPO 38.90 55.00 93.25 90.00 92.12 40.44 68.29 DeepSeek-R1-Distill-Qwen-7B + Self-Verification (Zhang et al., 2025a) 39.45 55.80 93.50 92.50 92.20 41.00 69.08 DeepSeek-R1-Distill-Qwen-7B + Critique-GRPO (Zhang et al., 2025b) 39.60 55.65 93.45 92.80 92.25 41.10 69.14 DeepSeek-R1-Distill-Qwen-7B + iGRPO (ours) 40.16 56.30 93.80 95.00 92.42 41.54 69.87 OpenMath-Nemotron-7B (Moshkov et al., 2025) 61.18 73.28 95.55 95.00 90.52 33.46 74.83 OpenMath-Nemotron-7B + GRPO 61.32 73.62 95.60 95.00 90.60 34.00 75.02 OpenMath-Nemotron-7B + iGRPO (ours) 62.45 74.79 96.00 97.50 90.75 34.94 76.07 DeepSeek-R1-Distill-Qwen-14B (Guo et al., 2025) 42.10 58.93 93.10 90.00 93.10 45.59 70.47 DeepSeek-R1-Distill-Qwen-14B + GRPO 43.70 60.26 93.10 91.20 93.50 46.00 71.29 DeepSeek-R1-Distill-Qwen-14B + iGRPO (ours) 45.52 64.06 94.00 93.45 94.00 47.06 73.02 OpenMath-Nemotron-14B (Moshkov et al., 2025) 61.18 73.28 95.55 95.00 94.01 33.46 75.41 OpenMath-Nemotron-14B + GRPO 64.53 74.79 96.00 95.00 94.40 35.70 76.73 OpenMath-Nemotron-14B + iGRPO (ours) 65.57 76.72 96.70 97.50 94.69 36.76 78.00

Let 𝐶gen denote the cost of producing one sampled completion (including prompt encoding, autoregressive decoding, and reward evaluation). Let 𝐺GRPO denote the number of completions sampled per prompt in standard GRPO.

Baseline GRPO Cost. Standard GRPO samples 𝐺GRPO completions per prompt, giving per-prompt rollout cost

𝐶GRPO ≈ 𝐺GRPO 𝐶gen.

iGRPO Rollout Cost. iGRPO samples 𝑁 Stage 1 drafts and 𝐺 Stage 2 refinements per prompt (Section 3.2), so

𝐶iGRPO 𝐶GRPO

𝐶iGRPO ≈ (𝑁 + 𝐺)𝐶gen,

In our experiments, we keep the sampling budget fixed by setting

=

𝑁 + 𝐺 𝐺GRPO

.

###### 𝑁 + 𝐺 = 𝐺GRPO,

so iGRPO redistributes the same number of rollouts across Stage 1 and Stage 2 rather than increasing them. For example, with 𝐺GRPO = 16 in GRPO, we use 𝑁 = 8 and 𝐺 = 8 in iGRPO, yielding

𝐶iGRPO ≈ (𝑁 + 𝐺)𝐶gen = 𝐺GRPO 𝐶gen ≈ 𝐶GRPO, i.e., the dominant generation cost is comparable to GRPO.

### 4. Experiments

#### 4.1. Setup

Our training data includes two datasets: MATH (Hendrycks et al., 2021) (7,500 step-by-step competition problems) and AceReason-Math (Chen et al., 2025b) (9,400 problems). All models are trained for one epoch,

with a KL divergence loss coefficient of 0 and no entropy regularization. We use a learning rate of 1×10−6 with a cosine schedule, generating eight completions per prompt (halved in each stage for iGRPO). The maximum prompt length is 1,024 tokens across all datasets. For all experiments, the completion length is capped at 4,096 tokens and the batch size at 1024. We evaluate model performance on well-known mathematical benchmarks, including AIME24/AIME25, MATH500, AMC23, GSM8K, and Minerva Math. For all benchmarks we report Pass@1 accuracy. However, for AIME24/AIME25 the reported value is averaged over 64 runs to ensure robustness. For other benchmarks, an average of 8 runs are reported. For evaluations, we use NeMo-Skills framework1 with decoding parameters such as a temperature of 0.6, top-p of 0.95, and generation length of 65,000. Additional training and evaluation details are provided in the supplementary materials.

#### 4.2. Results

Our goal in this section is to isolate the practical benefit of adding a self-feedback refinement stage to GRPO. Concretely, we ask whether the two-stage training signal in iGRPO translates into better verifiable mathematical reasoning, and whether those gains persist across model families and scales.

Controlled study with matched sampling budget. Table 1 compares iGRPO against vanilla GRPO and two recent self-improvement baselines, Self-Verification (Zhang

et al., 2025a) and Critique-GRPO (Zhang et al., 2025b), across model families and parameter scales (7B, 8B, 14B). All methods share the same training protocol and the same rule-based reward (Eq. 6). To ensure a fair compute comparison, we keep the total sampling budget fixed at eight completions per prompt for every method; iGRPO redistributes this budget across stages by using Stage 1 drafts to select self-feedback and Stage 2 completions to perform the GRPO-style update. As a result, differences in Table 1 reflect the effect of conditioning on the best draft rather than the effect of more sampling.

##### Generalist 8B model: the largest gains from self-feedback.

For Nemotron-H-8B-Base-8K, vanilla GRPO raises the macro-average from 29.65% to 41.08%. Self-Verification and Critique-GRPO further improve the average to 42.86% and 43.39%. iGRPO performs best at 45.04%, which is +3.96 points over GRPO and +1.65 points over the strongest self-improvement baseline. The improvements are most visible on the benchmarks that most strongly penalize near-miss reasoning, including AIME25 (9.17%) and Minerva (32.72%), and iGRPO also reaches the highest GSM8K accuracy (91.26%). This is a setting where avoiding additional self-judgment tasks matters: iGRPO does not require the model to generate critiques or verification rationales; it provides the best draft as a direct, high-signal scaffold and trains the policy to refine beyond it.

##### Stronger 7B distilled reasoner: consistent gains concentrated on multi-step tasks.

For DeepSeek-R1-Distill-Qwen-7B, the base model is already strong (61.93% average), and GRPO improves it to 68.29%. Self-Verification and Critique-GRPO reach 69.08% and 69.14%. iGRPO remains best overall at 69.87%. The gains concentrate on multi-step benchmarks where a mostly-correct attempt can fail due to a small late error (e.g., AIME24 at 56.30% and AMC at 95.00%). This pattern fits the iGRPO mechanism: Stage 1 increases the chance that a strong reasoning trajectory appears in the context, and Stage 2 learns to reliably “finish the job” under the same verifiable reward.

##### Math-specialized 7B model: improvements persist when the base is already strong.

For OpenMath-Nemotron-7B, the base model starts at 74.83% average. GRPO yields a small change to 75.02%, suggesting limited headroom from standard one-shot RL updates in this regime. iGRPO increases performance to 76.07%, with the most notable gains on the harder benchmarks, including AIME24 (from 73.28% to 74.79%) and AMC (from 95.00% to 97.50%). When drafts are already high-quality, the value of iGRPO is less about discovering a viable solution mode and more about systematically reinforcing the most reliable completion patterns.

1https://github.com/NVIDIA/NeMo-Skills

##### Scaling to 14B parameters: benefits persist on complex reasoning.

At the 14B scale, iGRPO continues to improve on GRPO across both model families. For DeepSeek-R1Distill-Qwen-14B, the macro-average increases from 71.29% (GRPO) to 73.02% (iGRPO), with a large gain on AIME24 from 60.26% to 64.06%. For OpenMath-Nemotron-14B, the macro-average improves from 76.73% (GRPO) to 78.00% (iGRPO), including gains on AIME25 from 64.53% to 65.57% and on AIME24 from 74.79% to 76.72%. While margins naturally shrink as models become stronger, the improvement pattern remains stable: iGRPO helps larger models correct residual errors that survive strong pretraining and standard RL fine-tuning, especially on long-horizon competition benchmarks.

##### Competitiveness against critique-style objectives.

Self-Verification and Critique-GRPO are strong baselines, but they ask the model to allocate capacity to additional behaviors (verifying, critiquing, or producing auxiliary text) that are only indirectly optimized by the outcome reward. iGRPO keeps the training loop tightly aligned with the verifiable objective while still capturing the benefit of iteration: Stage 1 supplies a high-quality draft as an explicit conditioning signal, and Stage 2 directly optimizes a refinement policy that can surpass that draft. Empirically, this design yields consistent improvements across model families and scales, with the strongest gains on benchmarks that are sensitive to small long-horizon reasoning mistakes.

- 4.3. Generalization to a Stronger Base and Harder Dataset All prior controlled comparisons were trained on the 7,500-problem MATH set. We

next evaluate iGRPO in a harder regime by both strengthening the initialization and shifting to a more challenging training distribution. We start from OpenReasoning-Nemotron-7B and train on AceReason-Math (Chen et al., 2025b), keeping the iGRPO configuration identical to Section 4;

OpenReasoning-Nemotron-7B

+ iGRPO

| |
|---|

| |
|---|

100

[Figure 3]

[Figure 4]

97.0

96.2

92.3

91.7

80

85.6

84.1

81.7

80.5

79.6

Pass@1(%)

77.9

72.8

71.9

60

62.9

61.1

- Figure 2 summarizes the results. iGRPO improves performance across all benchmarks, with the strongest gains on AIME24/25 (+1.52/+1.78) and clear transfer beyond math to GPQA (+1.84) and MMLU-Pro (+0.91), yielding a +1.23 overall average improvement. This indicates iGRPO remains effective with a stronger base model and harder training data, and that iterative selffeedback improves broadly useful refinement behaviors rather than only math-specific patterns.

40

20

0

GPQA MMLU-Pro AIME24 AIME25 MATH500 GSM8K Avg.

Figure 2: Pass@1 results for OpenReasoning-Nemotron7B with and without iGRPO. Improvements appear not only on math but also on general reasoning tasks such as MMLU-Pro and GPQA.

### 5. Ablation Beyond GRPO: self-feedback as a reusable refinement wrapper.

Our two-stage procedure is not tied to the GRPO objective itself: it can be layered on top of other group-based PPO variants by using Stage 1 to select a high-reward draft and Stage 2 to perform the base optimizer’s update on self-conditioned prompts. Table 2 applies this wrapper to DAPO and GSPO under matched rollout budgets (same total sampled completions per prompt, same reward, and identical training and evaluation settings). In both cases, self-feedback yields a consistent +1.1 to +1.2 macro-average improvement, indicating that the gains primarily stem from the refinement interface rather than GRPO-specific details.

Method Base Avg + iGRPO Avg Δ

DAPO 69.74 70.93 +1.19 GSPO 69.20 70.31 +1.11

Table 2: Beyond GRPO: addingthesameself-feedbackrefinement layer to other methods consistently improves the average Pass@1.

##### Generative judge study.

Because iGRPO only needs a scalar reward to (i) rank Stage 1 drafts and (ii) compute Stage 2 groupnormalized advantages, we can swap the binary outcome checker (Eq. 6) for a generative judge. On DeepSeek-R1-Distill-Qwen-7B trained on MATH, GPT-

- 5 scoring each solution in [0,1] improves Pass@1 on all six benchmarks and increases the average from 69.87 to 70.81 (+0.94; Table 3). The largest gains on AIME24/25 and Minerva are consistent with partial credit for near-miss traces, which lets them survive

- Stage 1 selection and be refined into correct answers in Stage 2.

Entropy analysis.

0 20 40 60 80 100

- 0.5
- 1

- 1.5
- 2

- 2.5

Training steps (%)

Entropy(nats)

iGRPO

GRPO

Figure 3: Entropy dynamics. Per-token policy entropy during training. iGRPO maintains higher mid-training entropy than GRPO, indicating sustained exploration before convergence.

To probe how self-feedback alters learning dynamics, we track the per-token Shannon entropy of the policy during RL. For a decoding step 𝑡 with context ℎ𝑡 and vocabulary 𝒱, we measure entropy in nats as

ℋ(︀

𝜋𝜃(· | ℎ𝑡)

)︀

= − ∑︁

𝑤∈𝒱

𝜋𝜃(𝑤 | ℎ𝑡) ln𝜋𝜃(𝑤 | ℎ𝑡),

(7) and in practice compute it from the log-softmax of the logits (base 𝑒), then average over all valid completion tokens in the batch to obtain a single scalar per training step. On DeepSeek-R1-Distill-Qwen-7B trained on MATH, both methods start at 2.45 nats, but GRPO collapses rapidly (0.60 at 10%, 0.42 by

30%) and then stays flat through the end. In contrast, iGRPO decays more gradually (0.80 at 15%, 0.48 at 30%) and remains slightly higher through mid-training (0.46 at 60%) before converging near GRPO (0.44 vs. 0.42 at 100%). This suggests iGRPO delays premature mode collapse: conditioning on the best draft encourages refinement around a strong scaffold while preserving alternative continuations long enough to recover from near-miss reasoning traces. Since final entropies are close, the gains are better explained by sustained mid-training exploration rather than higher randomness at convergence.

- 6. Conclusion

###### Benchmark Rule-based GPT-5 Judge Δ

AIME25 40.16 41.12 +0.96 AIME24 56.30 57.45 +1.15 MATH500 93.80 94.20 +0.40 AMC 95.00 96.25 +1.25 GSM8K 92.42 92.95 +0.53 Minerva 41.54 42.88 +1.34

Average 69.87 70.81 +0.94

Table 3: Effect of replacing the rule-based reward with a GPT-5 judge in iGRPO. Base model is DeepSeek R1 Distill Qwen 7B.

We introduced Iterative Group Relative Policy Optimization (iGRPO), a simple and effective extension of GRPO that injects an explicit self-feedback signal into outcome-driven RL for reasoning. iGRPO replaces single-shot optimization with a two-stage loop: Stage 1 samples multiple drafts and selects the highest-reward completion as feedback; Stage 2 conditions on this best draft and applies a standard GRPO-style update on refinements. This yields dynamic self-conditioning: the training context automatically improves as the policy improves, creating a bootstrapped refinement behavior that is absent from conventional group-based objectives. We also provided a clean theoretical characterization of the bootstrapping effect under binary rewards, showing that the expected quality of the selected draft increases monotonically with the policy’s success probability.

Empirically, iGRPO consistently improves verifiable math reasoning across model families and scales under matched rollout budgets. Across 7B, 8B, and 14B backbones, iGRPO outperforms vanilla GRPO as well as strong self-improvement baselines that rely on critique or verification behaviors (Table 1). In a harder generalization setting, training OpenReasoning-Nemotron-7B on AceReason-Math with iGRPO yields new best results on AIME24/AIME25 (85.62%/79.64%) and transfers gains beyond math to GPQA and MMLU-Pro (Figure 2). Our ablations further suggest that the benefit primarily comes from the refinement interface itself: the same two-stage wrapper improves other group-based PPO variants (Table 2), remains compatible with richer scalar rewards such as a generative judge (Table 3), and measurably alters learning dynamics by delaying premature entropy collapse (Figure 3).

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

AI-MO. Aimo validation aime dataset. https://huggingface.co/datasets/AI-MO/aimo-validation-aime,

- 2024a. 2

AI-MO. Aimo validation amc dataset. https://huggingface.co/datasets/AI-MO/aimo-validation-amc,

- 2024b. 2

Rosi Braidotti. A theoretical framework for the critical posthumanities. Theory, culture & society, 36(6):31–61,

2019. 1

Jiaqi Chen, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, and KwanYee K. Wong. Spc: Evolving self-play critic via adversarial games for llm reasoning, 2025a. URL https: //arxiv.org/abs/2504.19162. 3

Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025b. 2, 8, 10, 18

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/ forum?id=O4cHTxW9BS. 3

Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du, and Xiaolong Li. Self-playing adversarial language game enhances LLM reasoning. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. 3

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 2

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards. CoRR, abs/2502.01456, 2025. doi: 10.48550/ARXIV.2502.01456. URL https:

//doi.org/10.48550/arXiv.2502.01456. 1

John H Flavell. Metacognition and cognitive monitoring: A new area of cognitive–developmental inquiry. American psychologist, 34(10):906, 1979. 1

Linda Flower and John R Hayes. A cognitive process theory of writing. College Composition & Communication, 32(4):365–387, 1981. 1

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 2, 8

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874,

2021. 2, 8, 18

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Openreasoner-zero: An open source approach to scaling up reinforcement learning on the base model. CoRR, abs/2503.24290, 2025. doi: 10.48550/ARXIV.2503.24290. URL https://doi.org/10.48550/arXiv.2503.

###### 24290. 1

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 2

Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda. Prover-verifier games improve legibility of LLM outputs. CoRR, abs/2407.13692, 2024. doi: 10.48550/ARXIV.2407.13692. URL https://doi.org/10.48550/arXiv.2407.13692. 3

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. CoRR, abs/2411.15124, 2024. doi: 10.48550/ARXIV.2411.15124. URL https:

###### //doi.org/10.48550/arXiv.2411.15124. 2, 3

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022. 2

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023. 2

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783, 2025. doi: 10.48550/ ARXIV.2503.20783. URL https://doi.org/10.48550/arXiv.2503.20783. 1, 3

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023. 1

Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025. 2, 8, 17

NVIDIA. Openreasoning-nemotron-7b. https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B,

2025. 2 OpenCompass. Aime2025. https://huggingface.co/datasets/opencompass/AIME2025, 2025. MIT License. 2

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022. 1

George Polya. How to solve it: A new aspect of mathematical method. In How to solve it. Princeton university press, 2014. 1

###### Open R1. Openr1-math-220k. https://huggingface.co/datasets/open-r1/OpenR1-Math-220k, 2024. 2,

17 Donald A Schön. The reflective practitioner: How professionals think in action. Routledge, 2017. 1 John Schulman. Approximating kl divergence, 2020. URL http://joschu.net/blog/kl-approx.html. 4 Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,

YK Li, Y Wu, et al. DeepseekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652,

2023. 1

Herbert A Simon. The architecture of complexity. In The Roots of Logistics, pp. 335–361. Springer, 2012. 1 Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan

Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement learning for reasoning in large language models with one training example, 2025. URL https://arxiv. org/abs/2504.20571. 3

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing LLM reasoning with rule-based reinforcement learning. CoRR, abs/2502.14768, 2025. doi: 10.48550/ARXIV.2502.14768. URL https://doi.org/10.48550/arXiv.2502.

###### 14768. 2

Ziyu Ye, Rishabh Agarwal, Tianqi Liu, Rishabh Joshi, Sarmishta Velury, Quoc V. Le, Qijun Tan, and Yuan Liu. Evolving alignment via asymmetric self-play. CoRR, abs/2411.00062, 2024. doi: 10.48550/ARXIV.2411.

###### 00062. URL https://doi.org/10.48550/arXiv.2411.00062. 3

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025. doi: 10.48550/ARXIV.2503.14476. URL https://doi.org/10.48550/arXiv.2503.

###### 14476. 1, 3

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. URL https://arxiv. org/abs/2401.10020, 2024. 3

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. CoRR, abs/2503.18892,

2025. doi: 10.48550/ARXIV.2503.18892. URL https://doi.org/10.48550/arXiv.2503.18892. 1 Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, and Bo An. Incentivizing llms to self-verify their answers. arXiv preprint arXiv:2506.01369, 2025a. 3, 8, 9

Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025b. 3, 8, 9

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 3

Appendix

- A. Policy Gradient Derivation for iGRPO

In this sectin, we derive the iGRPO policy gradient update consistent with the methodology in Section 3.2 and the full objective in Eq. equation 5. Throughout, 𝜋𝜃 is the trainable policy, 𝜋𝜃

is the frozen snapshot used for sampling (PPO/GRPO convention), 𝜋ref is the reference policy, and 𝑅𝜑(·) is a scalar (verifier) reward. Importantly, iGRPO uses two stages per optimization step, but gradients are applied only to Stage 2 tokens.

old

#### A.1. Two-stage sampling and the induced self-conditioned prompt distribution

For each prompt 𝑞 ∼ 𝑃(𝑄), iGRPO constructs an augmented prompt by first sampling Stage 1 drafts and selecting the best:

(· | 𝑞), 𝑖 = 1,...,𝑁, (8) 𝑑ˆ= arg max

𝑑𝑖 ∼ 𝜋𝜃

old

𝑅𝜑(𝑑𝑖), (9)

𝑖∈{1,...,𝑁}

𝑞′ = Concat(𝑞,𝑑ˆ). (10) This defines an implicit sampling distribution over augmented prompts 𝑞′ induced by 𝜋𝜃

and the arg max selection. Within a single PPO-style update, 𝑞′ is treated as part of the sampled context (no differentiation through Stage 1 and no differentiation through the arg max). Across iterations, the distribution over 𝑞′ shifts because 𝜋𝜃

old

changes. Given the augmented prompt 𝑞′, Stage 2 samples a group of 𝐺 completions:

old

(· | 𝑞′), 𝑗 = 1,...,𝐺. (11) Let each completion be a token sequence 𝑜𝑗 = (𝑜𝑗,1,...,𝑜𝑗,|𝑜

𝑜𝑗 ∼ 𝜋𝜃

old

𝑗|) with factorization

∏︁|𝑜𝑗|

𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (12)

𝜋𝜃(𝑜𝑗 | 𝑞′) =

𝑡=1

#### A.2. From the self-conditioned expected reward to a REINFORCE-style gradient Fix an augmented prompt 𝑞′. Consider the (conceptual) self-conditioned objective

[︀

]︀

. (13) By the score-function (REINFORCE) identity,

𝒥 (𝜃 | 𝑞′) = E𝑜∼𝜋

𝑅𝜑(𝑜)

𝜃(·|𝑞′)

𝜃(·|𝑞′)[︁𝑅𝜑(𝑜)∇𝜃 log 𝜋𝜃(𝑜 | 𝑞′)]︁ (14)

∇𝜃𝒥 (𝜃 | 𝑞′) = E𝑜∼𝜋

[︃𝑅𝜑(𝑜)

∇𝜃 log 𝜋𝜃(𝑜𝑡 | 𝑞′,𝑜<𝑡)]︃, (15)

∑︁|𝑜|

= E𝑜∼𝜋

𝜃(·|𝑞′)

𝑡=1

where the second line uses Eq. equation 12. Introducing any baseline 𝑏(𝑞′) that does not depend on the sampled tokens preserves unbiasedness and yields an advantage-weighted gradient:

∇𝜃 log 𝜋𝜃(𝑜𝑡 | 𝑞′,𝑜<𝑡)]︃. (16)

[︃

)︀∑︁|𝑜|

(︀

∇𝜃𝒥 (𝜃 | 𝑞′) = E𝑜∼𝜋

𝑅𝜑(𝑜) − 𝑏(𝑞′)

𝜃(·|𝑞′)

𝑡=1

#### A.3. Group-relative advantage used in iGRPO

In iGRPO, the baseline is estimated per prompt by sampling a group of 𝐺 completions {𝑜𝑗}𝐺𝑗=1 and normalizing rewards within the group. Define

𝑅𝑗 = 𝑅𝜑(𝑜𝑗), 𝑅 = mean({𝑅1,...,𝑅𝐺}), 𝑠𝑅 = std({𝑅1,...,𝑅𝐺}), (17)

and the iGRPO advantage

𝑅𝑗 − 𝑅 𝑠𝑅

, with the convention 𝐴ˆ𝑗 = 0 if 𝑠𝑅 = 0. (18)

𝐴ˆ𝑗 =

As in GRPO, this advantage is a single scalar per completion, shared across all token indices. iGRPO additionally uses the token-averaged form (the 1/|𝑜𝑗| factor) so that completions of different lengths contribute comparably:

∑︁|𝑜𝑗|

∑︁𝐺

𝐴ˆ𝑗 |𝑜𝑗|

1 𝐺

. =

∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (19)

𝑔on-policy(𝜃 | 𝑞′)

𝑡=1

𝑗=1

Eq. equation 19 is the basic (conceptual) iGRPO policy-gradient estimator when sampling from 𝜋𝜃. The remainder of the derivation follows the standard PPO/GRPO stabilization used in the main methodology: sampling from 𝜋𝜃

and using importance ratios with clipping and a KL penalty.

old

#### A.4. Off-policy sampling from 𝜋𝜃

and PPO-style clipping

old

Within each iteration, iGRPO samples Stage 2 completions from 𝜋𝜃

for stability. To relate gradients under 𝜋𝜃

old

to the updated policy 𝜋𝜃, define the per-token importance ratio

old

𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃

. (20)

𝑟𝑗,𝑡(𝜃) =

(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡)

old

Following PPO/GRPO, iGRPO maximizes a clipped surrogate that replaces the on-policy factor by a clipped importance-weighted term:

ℒclip𝑗,𝑡 (𝜃) = min(︁𝑟𝑗,𝑡(𝜃)𝐴ˆ𝑗, clip

𝐴ˆ𝑗)︁. (21)

(︀

)︀

𝑟𝑗,𝑡(𝜃), 1 − 𝜖, 1 + 𝜖

The per-token clipped objective in Eq. equation 21 yields the usual piecewise gradient behavior: when the ratio is clipped, the clipped branch is constant in 𝑟𝑗,𝑡(𝜃) and contributes zero gradient through that branch.

- A convenient way to write the gradient is via the indicator of the unclipped branch. Let

⎧ ⎪⎨

1, 𝐴ˆ𝑗 ≥ 0 and 𝑟𝑗,𝑡(𝜃) ≤ 1 + 𝜖, 1, 𝐴ˆ𝑗 < 0 and 𝑟𝑗,𝑡(𝜃) ≥ 1 − 𝜖, 0, otherwise,

(22)

I𝑗,𝑡(𝜃) =

⎪⎩

which matches the standard PPO clipping rule. Then, using ∇𝜃𝑟𝑗,𝑡(𝜃) = 𝑟𝑗,𝑡(𝜃)∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡), the gradient of the clipped surrogate is

∇𝜃ℒclip𝑗,𝑡 (𝜃) = I𝑗,𝑡(𝜃) 𝐴ˆ𝑗 𝑟𝑗,𝑡(𝜃) ∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (23)

#### A.5. Including the per-token KL penalty

As in Section 3.1, iGRPO includes a per-token KL penalty to a reference policy 𝜋ref using the non-negative estimator

𝜋ref(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) − 1. (24)

𝜋ref(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡) − log

𝐷̂︀KL(𝑗,𝑡) =

Let 𝜌𝑗,𝑡(𝜃) = 𝜋ref(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡)/𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). Then 𝐷̂︀KL(𝑗,𝑡) = 𝜌𝑗,𝑡 − log 𝜌𝑗,𝑡 − 1, and its gradient has a simple form:

)︀ ∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (25) Therefore, the KL-regularized contribution in the objective, −𝛽 𝐷̂︀KL(𝑗,𝑡), contributes

∇𝜃𝐷̂︀KL(𝑗,𝑡) = −(︀

𝜌𝑗,𝑡(𝜃) − 1

(︀

)︀ ∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (26)

−𝛽 ∇𝜃𝐷̂︀KL(𝑗,𝑡) = 𝛽

𝜌𝑗,𝑡(𝜃) − 1

- Table S.1: Performance comparison of OpenMath-Nemotron-14B and iGRPO-enhanced variant.

Model AIME25 AIME24 MATH500 AMC GSM8K Minerva Avg

OpenMath-Nemotron-14B (Moshkov et al., 2025) 61.18 73.28 95.55 95.00 94.01 33.46 75.41 OpenMath-Nemotron-14B + iGRPO 66.04 76.61 96.90 97.50 94.16 38.24 78.24

#### A.6. Final iGRPO surrogate objective and resulting policy gradient

Putting the pieces together and reinstating the full two-stage sampling (where Stage 1 affects the distribution of 𝑞′ but is not differentiated through within an iteration), the iGRPO surrogate objective is:

𝒥iGRPO(𝜃) = E[︁𝑞 ∼ 𝑃(𝑄)]︁ E[︁{𝑑𝑖}𝑁𝑖=1 ∼ 𝜋𝜃

𝑅𝜑(𝑑𝑖), 𝑞′ = Concat(𝑞,𝑑ˆ), {𝑜𝑗}𝐺𝑗=1 ∼ 𝜋𝜃

(· | 𝑞), 𝑑ˆ= arg max

old

𝑖

(· | 𝑞′)]︁

old

∑︁|𝑜𝑗|

[︁ℒclip𝑗,𝑡 (𝜃) − 𝛽 𝐷̂︀KL(𝑗,𝑡)]︁, (27)

∑︁𝐺

1 𝐺

1 |𝑜𝑗|

×

𝑡=1

𝑗=1

where ℒclip𝑗,𝑡 (𝜃) is defined in Eq. equation 21, 𝑟𝑗,𝑡(𝜃) in Eq. equation 20, and 𝐷̂︀KL(𝑗,𝑡) in Eq. equation 24. This matches the structure given in Eq. equation 5.

Differentiating Eq. equation 27 yields the iGRPO policy gradient:

∑︁|𝑜𝑗|

∇𝜃𝒥iGRPO(𝜃) = E[︁···]︁ 1 𝐺

[︁∇𝜃ℒclip𝑗,𝑡 (𝜃) − 𝛽 ∇𝜃𝐷̂︀KL(𝑗,𝑡)]︁, (28)

∑︁𝐺

1 |𝑜𝑗|

𝑡=1

𝑗=1

with the explicit per-token forms from Eqs. equation 23 and equation 25. Concretely, combining them gives

∑︁|𝑜𝑗|

[︁I𝑗,𝑡(𝜃)𝐴ˆ𝑗 𝑟𝑗,𝑡(𝜃) + 𝛽

∇𝜃𝒥iGRPO(𝜃) = E[︁···]︁ 1 𝐺

)︀]︁ ∇𝜃 log 𝜋𝜃(𝑜𝑗,𝑡 | 𝑞′,𝑜𝑗,<𝑡). (29)

∑︁𝐺

(︀

1 |𝑜𝑗|

𝜌𝑗,𝑡(𝜃) − 1

𝑡=1

𝑗=1

##### Interpretation.

Eq. equation 29 makes the roles of the two stages explicit. Stage 1 defines the self-conditioned context 𝑞′ (and thus the learning problem presented to Stage 2) but does not contribute direct gradients within an iteration.

- Stage 2 contributes a standard GRPO/PPO-style token-level policy gradient, where each token is weighted by a group-normalized advantage 𝐴ˆ𝑗 (shared across tokens of the completion), stabilized by importance-ratio clipping, and regularized by a KL penalty to 𝜋ref.

### B. Scaling OpenMath-Nemotron-14B with iGRPO

To further validate the effectiveness of iGRPO, we trained the OpenMath-Nemotron-14B with iGRPO on large scale dataset of OpenR1-Math-220k (R1, 2024) which consists of 220,000 math problems with reasoning traces from DeepSeek-R1. For this study, we use 94,000 examples. As shown in Table S.1, the reasoning performance of the model trained with iGRPO is significantly improved, achieving an impressive AIME25 score of 66.04%.

#### B.1. Analysis of Pass@N on AIME Benchmarks

Figure S.1 shows how the accuracy of iOpenMath-Nemotron-14B evolves when increasing the number of attempts 𝑁 on the AIME24 and AIME25 benchmarks. As expected, we see consistent gains in performance as 𝑁 grows, indicating that the model can generate correct solutions among multiple sampled responses even if the top-1 guess is sometimes incorrect. Although our base SFT model already shows strong results at pass@1, the additional RL fine-tuning appears to capitalize on multi-sample scenarios. For example, on AIME25, the model improves from 66.04% (pass@1) to 86.67% at pass@8, demonstrating the effectiveness of producing multiple solutions for challenging competition problems.

AIME Benchmarks: Accuracy vs pass@N

AIME25 AIME24

95

90

Accuracy(%)

85

80

75

70

65

1 2 4 8 16 32 64 128 256

pass@N

- Figure S.1: Performance of iOpenMath-Nemotron-14B across various pass@N settings for AIME24 and AIME25. Both benchmarks exhibit increasing accuracy with higher 𝑁, though AIME24 quickly stabilizes at 93.33% by 𝑁 = 16, whereas AIME25 continues to rise until reaching 96.67% at 𝑁 = 256.

Despite these gains, there are clear saturation points. AIME24 converges to its best score of 93.33% by 𝑁 = 16, with no further improvement at higher 𝑁. On the contrary, the performance of AIME25 continues to improve even at high values of 𝑁. While the performance seems to plateau briefly at 90.00% between 𝑁 = 32 and 𝑁 = 128, it eventually increases to 96.67% at 𝑁 = 256. This contrast suggests that certain problem distributions, especially in AIME25, may benefit from a larger number of sampled attempts, whereas others, such as those in AIME24, can be adequately solved with fewer solution attempts.

### C. Hyperparameter Setup

We conduct ablation studies on both 7B and 14B model variants, spanning core architectures such as DeepSeekR1-Distill-Qwen and OpenMath-Nemotron. All models are trained for one epoch across two datasets: (1) the MATH dataset (Hendrycks et al., 2021) of 7,500 step-by-step problems, and (2) AceReason-Math (Chen et al., 2025b) dataset (9400 problems). Our training uses a KL divergence loss coefficient of 0. We set a cosine learning rate schedule (minimum rate of 0.1) with a base learning rate of 1 × 10−6. We use 8 rollouts for all experiments.

Table S.2 lists the concrete hyperparameters for training 7B iGRPO models. Notably, we run on 2 nodes with 8 × NVIDIA A100 GPUs each, and one of these nodes is fully allocated to vLLM for generation. We keep a global batch size of 128, with a per-device batch size of 16 and a gradient accumulation step size of 8. For the 14B models, we scale out to 5 nodes of 8 × NVIDIA A100 GPUs each (again, one node reserved for vLLM) and reduce the per-device batch size to 4 (maintaining the same global batch size of 128). In both 7B and 14B setups, we continue to use bfloat16 precision and the FlashAttention-2 kernel. The temperature is set to 0.7 for generation, and we apply two reward functions (accuracy and format) each with weight 1.0. This configuration provides a balanced trade-off between training stability, throughput, and alignment with complex mathematical reasoning tasks.

Prompt: We use the following prompt for training model with iGRPO.

- Table S.2: Hyperparameters for training 7B models with iGRPO using 2 nodes each using 8 × NVIDIA A100 GPUs. One node is entirely reserved for vllm.

##### Parameter Value

bf16 true attn_implementation flash_attention_2 use_vllm true vllm_gpu_memory_utilization 0.85 gradient_accumulation_steps 8 gradient_checkpointing true learning_rate 1e-06 lr_scheduler_type cosine_with_min_lr lr_scheduler_kwargs min_lr_rate: 0.1 warmup_ratio 0.1 num_train_epochs 1 per_device_train_batch_size 16 max_completion_length 4096 num_generations 4 temperature 0.7 reward_funcs accuracy, format reward_weights 1.0, 1.0

Prompt

You are a helpful AI Assistant that provides well-reasoned and detailed responses. First think through your reasoning as an internal monologue, then give the answer: <think>...</think><answer>...</answer>. If the prompt contains feedback or a prior draft, treat it as guidance, not as ground truth. Use it to produce a strictly improved final answer: fix mistakes, fill gaps, strengthen justification, and improve clarity. Do not repeat the draft verbatim. If the feedback is wrong or incomplete, correct it and proceed.

### D. Memory and Throughput Comparisons

#### D.1. Setup

To evaluate the resource utilization of our training setup, we replicate the exact environment and conditions under which our 7B models are typically trained. Specifically, we use DeepSeek-R1-Distill-Qwen-7B as our base model, which serves as a representative checkpoint for measuring throughput and memory consumption when training on the MATH dataset. Our training configuration employs a per-device batch size of 16, along with a global gradient accumulation step of 8, allowing us to effectively simulate heavier loads without exceeding GPU memory constraints. Additionally, we use a maximum completion length of 2048 tokens to benchmark model performance. We run experiments on two nodes, each equipped with 8 × NVIDIA A100 GPUs. One node is dedicated to vllm generation, ensuring that inference or generation processes do not interfere with the primary training workload, while the other node is reserved exclusively for model training.

We measure peak memory usage by periodically querying the GPU memory allocator for the maximum memory allocation that has occurred since the start of training. Specifically, at the beginning of training, we reset the peak memory statistics, and then after each iteration, we retrieve the current peak memory usage in bytes. We convert this value to gigabytes for readability and log it alongside other training metrics. To measure throughput, we track the total number of samples processed over time. We calculate this by multiplying the current global step by both the per-device batch size and the number of devices used in data parallelism. Dividing this product by the elapsed training time in seconds yields the throughput, expressed as samples

processed per second. This real-time monitoring of memory and throughput allows us to evaluate hardware utilization efficiency, identify possible bottlenecks, and compare different training configurations in a consistent and quantifiable manner.

#### D.2. Measurements

- Table S.3 presents measured GPU usage and training throughput under iGRPO vs GRPO. Despite the two-stage nature of iGRPO, its peak memory usage of 54.9349GB closely matches GRPO’s 54.9286GB, a difference of roughly 0.0063GB, which is practically negligible. This matches our theoretical expectation that the selffeedback mechanism adds minimal overhead, validating the feasibility of integrating iterative refinements even under constrained resource budgets.

Regarding throughput, iGRPO processes 0.34samples/s compared to GRPO’s 0.41samples/s, reflecting a mild slowdown tied to the additional round of generation. Crucially, this is neither an order-of-magnitude nor a large factor reduction. Instead, it shows that iGRPO’s second-stage refinement imposes only a modest computational cost. In summary, these measurements confirm our claims that iGRPO can be implemented with little additional overhead, supporting it as a practical strategy for enhancing mathematical reasoning performance without compromising resource efficiency.

Beyond instantaneous throughput, we also report full training cost measured in total GPU hours. Under the same compute budget and eight generations per prompt, GRPO requires 83.3 GPU hours while iGRPO uses 94.1 GPU hours, which corresponds to roughly a 13% increase in wall-clock training time. This overhead arises from the sequential Stage 1 plus Stage 2 decoding but does not demand more GPUs or additional memory capacity, since peak usage remains essentially unchanged. Given that this modest time increase delivers several-point gains on AIME24 and AIME25 and enables our 7B models to reach state-of-the-art performance, we view the tradeoff between 13% extra training time and substantially higher reasoning accuracy as a favorable and practical value proposition in real deployments.

Method Peak Memory (GB) Throughput (Samples/sec) Total GPU Hours

GRPO 54.9286 0.41 83.3 iGRPO 54.9349 0.34 94.1

- Table S.3: Empirical memory usage and throughput on an 80GB A100 setup. iGRPO’s two-stage approach yields near-identical peak memory usage and only a minor decrease in throughput compared to GRPO. The last column reports total GPU hours for a full training run, showing that iGRPO adds only a modest 13% time overhead relative to GRPO.

### E. Additional Ablation Studies

##### Training Dynamics and Response Length.

As shown in Fig. S.2, we compare average rewards for iGRPO and GRPO at multiple checkpoints, observing that iGRPO consistently maintains a higher reward throughout training. The iterative refinement in iGRPO ultimately yields a superior reward trajectory. In addition, we measure the response length over training steps and find that both methods exhibit nearly identical lengths, with GRPO producing slightly longer outputs on average. Notably, iGRPO’s two-stage process does not manifest in lengthy completions but instead appears to refine solutions within a similar token budget. This indicates that the gains from iterative refinement arise more from improved response quality than from verbosity.

##### Effect of KL Divergence Term.

We vary the coefficient 𝛽 ∈ {0,0.0001,0.001,0.01} to examine how tightly the policy is regularized against the reference model. As shown in Table S.4, while 𝛽 = 0.0001 achieves the highest overall score (70.23%), the difference among all settings is relatively small. The KL term, in principle, balances exploration with adherence to the current policy. However, given the marginal gains observed, setting 𝛽 = 0 offers a simpler training pipeline without sacrificing significant performance. Hence, we use 𝛽 = 0 to reduce overhead and maintain efficiency.

0.7

###### AverageReward

0.6

0.5

0.4

0.3

0.2

iGRPO

0.1

GRPO

0.0

0 30 60 90 120 150

Training steps

(a) Average training reward.

###### ResponseLength

900

800

700

600

iGRPO

500

GRPO

400

0 30 60 90 120 150

Training steps

(b) Response length.

- Figure S.2: Comparison of (a) average training rewards and (b) response lengths for GRPO vs. iGRPO.

- Table S.4: Ablation results for iGRPO. We study the effect of the KL divergence coefficient 𝛽 (top) and the total number of completions (bottom).

Effect of KL Divergence Coefficient 𝛽 𝛽 Score (%) 0 69.87

0.0001 70.23 0.001 69.31

0.01 69.91 Effect of Number of Completions Total Completions Score (%)

4 67.79 8 69.87

16 70.17 32 70.33

##### Effect of Number of Completions.

We study how the total number of completions in iGRPO affects performance, allocating 4,8,16, or 32 completions evenly across the two stages. As shown in Table S.4, increasing from 4 to 8 completions gives a clear improvement, while gains beyond 8 are modest. Larger budgets also increase training time and inference latency for minimal returns.

