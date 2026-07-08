## arXiv:2604.27039v1[cs.CL]29Apr2026

# Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling

Zhen Zhang1, Changyi Yang2,4, Zijie Xia4, Zhen Yang5, Chengzhi Liu1, Zhaotiao Weng1, Yepeng Liu1, Haobo Chen1, Jin Pan3,4, Chenyang Zhao4, Yuheng Bu1, Alkesh Patel5, Zhe Gan5, Xin Eric Wang1

1University of California, Santa Barbara, 2Carnegie Mellon University, 3University of Wisconsin–Madison, 4LMSYS Org, 5Apple Inc.

Token serves as the fundamental unit of computation in modern autoregressive models, and generation length directly influences both inference cost and reasoning performance. Despite its importance, existing approaches lack fine-grained length modeling, operating primarily at the coarse-grained sequence level. In this paper, we introduce the Length Value Model (LenVM), a token-level framework that models the remaining generation length at each decoding step. By formulating length modeling as a value estimation problem and assigning a constant negative reward to each generated token, LenVM predicts a bounded, discounted return that serves as a monotone proxy for the remaining generation horizon. This formulation yields supervision that is annotation-free, dense, unbiased, and scalable. Experiments on LLMs and VLMs demonstrate that LenVM provides a highly effective signal at inference time. On the LIFEBench exact length matching task, applying LenVM to a 7B model improves the length score from 30.9 to 64.8, significantly outperforming frontier closed-source models. Furthermore, LenVM enables continuous control over the trade off between performance and efficiency. On GSM8K at a budget of 200 tokens, LenVM maintains 63 percent accuracy compared to 6 percent for token budget baseline. It also accurately predicts total generation length from the prompt boundary. Finally, LenVM’s token-level values offer an interpretable view of generation dynamics, revealing how specific tokens shift reasoning toward shorter or longer regimes. Results demonstrate that LenVM supports a broad range of applications, including length control, prediction, and interpretation of generation dynamics. They suggest that generation length can be effectively modeled as a token-level value signal, highlighting the potential of LenVM as a general framework for length modeling and as a length-specific value signal that could support future RL training.

[Figure 1]

[Figure 2]

Correspondence: zhen_zhang@ucsb.edu, ericxwang@ucsb.edu Project Page: https://github.com/eric-ai-lab/Length-Value-Model

###### Token-level Value Supervision

###### Scalability Axes

Remaining Horizon −

L L-1 L-2 L-t 1

0

Model Size # Prompts # Completions

[Figure 3]

[Figure 4]

[Figure 5]

Discounted Returns

Proxy

[Figure 6]

[Figure 7]

Regression Token-wise MSE Loss

[Figure 8]

Loss decreases consistantly with scale

###### LenVM

Value Head

[Figure 9]

###### Data Generation

- Completion 1
- Completion 2

…

Final Layer Hidden State

[Figure 10]

Prompts

LLM / VLM

…

[Figure 11]

Model

…

…

…

Last Token

Completion n

Sample n completions per prompt

Steps 1 2 L-1

0

L

…

Completion

Prompt

Figure 1. Length Value Model (LenVM) architecture and Pipeline. At each decoding step, LenVM attaches a scalar value head to the final-layer hidden state of an LLM or VLM to predict the value Vt ∈ (−1,0). LenVM enables annotation-free and scalable value pretraining across model size, prompt count, and number of completions.

1

### Contents

- 1 Introduction 3
- 2 Related Work 4

- 2.1 Length-Controlled Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Output Length Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 RL Framing and Reward Shaping of Generation Length . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Length Value Model 5

- 3.1 Modeling Length as a Discounted Return . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.2 LenVM Head . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.3 Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4 Experiments: Validating LenVM as a Token-Level Length Signal 7

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Application 1: Length-Controlled Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.3 Performance–Efficiency Trade-off . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.4 Application 3: Generation Length Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.5 Scalability of LenVM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.6 Case Study: Length Tokens as Markers of Length Shifts . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5 Ablations 12

- 5.1 Length-Space Representation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.2 Batch Construction Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.3 Discount Factor γ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.4 Numerical Precision . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 6 Conclusion 14 Appendix 18

- A Additional Experimental Details 18
- B LIFEBench Evaluation Details 18
- C LenVM as a Length-Specific Value Function in RL 19

- C.1 LenVM as a Value Function for a Length Objective . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Advantage Decomposition and Policy Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 LenVM as a Potential for Policy-Invariant Reward Shaping . . . . . . . . . . . . . . . . . . . . . . 21
- C.4 Why This is Preferable to a Sequence-Level Length Penalty . . . . . . . . . . . . . . . . . . . . . . 22
- C.5 Connection to the Main Paper . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Finite-Precision Analysis of Relative Length Resolution 23
- E Future-Dependent Weighting Changes the Regression Target 24

- E.1 General Weighted Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- E.2 Instantiations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- F Why Inverting the Transformed Horizon Underestimates Expected Remaining Length 26
- G Derivation of the Exponential Tilting Solution in the Performance–Efficiency Trade-off Experiment 27

#### 1. Introduction

Token generation is the core computational process in modern AI systems. Large language models (LLMs), vision-language models (VLMs), and specialized models for code, mathematics, and agentic tasks all produce outputs sequentially. The number of generated tokens therefore forms a fundamental unit of inferencetime computation. It influences both performance, since additional tokens can enable more reasoning in challenging settings (Snell et al., 2024), and cost, since each token incurs compute, KV-cache memory, latency, and energy (Kwon et al., 2023, Pope et al., 2022). Despite its importance, existing approaches to length modeling and control remain coarse-grained. Sequence-level penalties during training (Team et al., 2025), prompt-based instructions at inference (Xiao et al., 2026a), and pre-decode predictors (Zhang et al., 2025a) all operate at the level of whole sequences or fixed pre-generation decisions, rather than the token-level dynamics of generation itself. What is missing is a token-level model of remaining generation length.

To address this gap, we propose the Length Value Model (LenVM), a token-level formulation of generation length modeling. At each decoding step, LenVM predicts a scalar value for the current decoding state. Conceptually, the quantity of interest is the remaining generation length, which depends on how much future generation is still to be realized. We model this quantity through a value formulation: by assigning a constant negative reward to each generated token and discounting future steps by γ, we define a discounted return that serves as a bounded, monotone proxy for the remaining generation horizon. LenVM is trained to predict this return, yielding a principled reduction from token-level length modeling to value estimation and placing the problem in a standard value-function framework. In this view, LenVM evaluates a decoding state by its remaining distance to completion rather than by task utility. This construction also maps highly variable raw lengths into a bounded target space, improving conditioning when generation lengths range from a few tokens to over 32k. The resulting value signal can be used at inference time for length estimation and control, and at training time as a length-aware value baseline.

A central advantage of LenVM is that it naturally supports scalable value pretraining. The resulting supervision signal has four favorable properties. First, it is annotation-free: training targets are constructed automatically from sampled completions, without any additional human labeling or reward modeling. Second, it is dense: every token position contributes a target, rather than only one target per completed response. Third, it is unbiased for the policy-conditioned value target: under a fixed rollout policy, the realized return is an unbiased Monte Carlo sample of the target value and introduces no additional annotation or reward-model noise. Fourth, it is scalable: supervision grows naturally with both prompt count and the number of completions sampled per prompt, while remaining cheap to construct and well matched to increasing model size. Together, these properties make LenVM well suited to large-scale value pretraining, enabling high-quality token-level value supervision without the annotation bottlenecks of conventional reward and value modeling.

We evaluate whether LenVM learns a meaningful token-level signal of remaining generation length. At inference time, LenVM exposes a smooth performance–efficiency trade-off: exponentially tilting the token distribution with predicted values traces a Pareto frontier between response quality and generation length. The same signal also supports explicit length control, where LenVM-guided decoding improves adherence to hard constraints such as Equal To, At Most, and At Least on LIFEBench (Zhang et al., 2025b) (§ 4.2, § 4.3)). In addition, LenVM provides useful length prediction, including from the very first generated token, across math, code, and instruction-following domains (§ 4.4). Its token-level values further offer an interpretable view of generation dynamics, revealing where trajectories shift between shorter and longer reasoning regimes (§ 4.6). Beyond inference, we show that LenVM admits a natural interpretation as a length-specific value baseline in PPO-style RL (Schulman et al., 2017), and we provide scaling evidence

that its value-learning objective improves consistently with scale (§ 4.5). Together, these results establish LenVM as a principled and scalable framework for token-level length modeling, while leaving empirical RL fine-tuning to future work.

#### 2. Related Work

###### 2.1. Length-Controlled Generation

Length control has been explored through prompting, fine-tuning, and decoding-time constraints. Promptbased methods can be attractive for black-box models: for example, Xie and Lee propose a one-shot countdown prompt that improves strict compliance without changing model weights (Xie and yi Lee, 2025). Recent prompt-only methods also incorporate explicit structure and counting to regulate length without retraining; Plan-and-Write uses a plan-first scaffold and word-counting guidance to improve adherence in summarization tasks (Akinfaderin et al., 2025). For black-box settings where fine-tuning is not possible, Gu et al. (2024) propose an iterative sampling framework that combines Metropolis–Hastings with importancesampling acceleration to achieve length-constrained generation without modifying model parameters. More broadly, constrained generation has a long history of MCMC-style inference-time methods; CGMH applies Metropolis–Hastings moves to satisfy lexical constraints without parallel training data (Miao et al., 2018), and recent work revisits MCMC constrained sampling with stronger distributional guarantees (Gonzalez et al., 2025). Training-based approaches can yield more precise control by modifying the model’s internal length representation; Butcher et al. (2024) introduce length-difference positional encoding (LDPE) and fine-tuning to encourage coherent termination near the target length , while Xie et al. (2026) propose Hansel, which uses periodically outputted hidden special tokens to track the remaining target length during finetuning. In contrast to approaches that embed length information into the generator, our method trains a separate token-level horizon estimator and uses it as a decoding-time control signal, enabling length control without retraining the base LLM.

###### 2.2. Output Length Prediction

Predicting completion length is important for both serving efficiency and stochastic sampling, and can be approached either statically (before decoding) or progressively (during decoding). Recent work shows that hidden representations and entropy signals can provide accurate, low-overhead length prediction; in particular, Xie et al. (2026) propose entropy-guided token pooling for static prediction and progressive length prediction (PLP) for online remaining-length estimation . Beyond entropy-guided methods, Piotrowski et al. (2025) study forecasting remaining output length from frozen layerwise hidden states, exploring aggregation and graph-based regressors . In code generation, horizon-length prediction has also been used to improve fill-in-the-middle planning: Ding et al. (2025) propose a horizon-length prediction objective to better align infills with distant right context. Relatedly, Xiao et al. (2026b) examine whether LLMs can self-track output length and propose a dynamic feedback mechanism that adjusts generation online to meet length targets without retraining. Our LVM is also an online predictor, but is trained as a standalone value model with bounded regression targets and is designed to provide a stable per-token signal across diverse task families and length scales. Variable output lengths complicate batching and can waste compute through padding and fragmentation; length prediction can improve schedulers and load balancing. Zheng et al. (2023) study response length perception and sequence scheduling to improve inference throughput.

###### 2.3. RL Framing and Reward Shaping of Generation Length

Our formulation views length regularization as a dense per-step penalty (i.e., negative reward) with an associated discounted return, making remaining horizon a value-function quantity. This makes LVM naturally compatible with PPO-style training as a length-specific value baseline and advantage component (Section C), while in this paper we focus on inference-time decoding control and analysis. More broadly, recent RLHF work argues for value-centric formulations: DVPO pretrains a global value model on preference data and uses it as a frozen critic for policy optimization (Huang et al., 2026), and V0 learns a generalist value estimator at the initial prompt for policy-agnostic routing and resource allocation (Zhang et al., 2026). Recent work on efficient reasoning also studies how to adapt length penalties during RL to balance accuracy and conciseness (Xiang et al., 2025, Liu et al., 2025, Li et al., 2025). VAPO further targets stable and efficient value-based RL for long-CoT reasoning by addressing value bias, heterogeneous sequence lengths, and sparse rewards (Yue et al., 2025).

#### 3. Length Value Model

We propose the Length Value Model (LenVM), a token-level value function over decoding states for modeling generation length. As shown in Figure 1, LenVM treats autoregressive generation as an episodic process and models remaining generation length through a discounted return induced by a constant per-token cost. Rather than predicting raw remaining length directly, it estimates a bounded and monotone transform of the remaining generation horizon. Starting from the last prompt token, LenVM outputs a scalar value at each decoding step. We first define the return construction (§3.1), then summarize its key properties, and finally describe the prediction head (§3.2) and training objective (§3.3).

###### 3.1. Modeling Length as a Discounted Return

Let st denote the decoding state after the prompt and the first t generated tokens, with s0 corresponding to the last prompt token. For a sampled completion of generated length L, decoding steps are indexed by

t ∈ {0, . . . , L}, where t = L is the EOS step. Our goal is to estimate, from each prefix state st, how much generation horizon remains under the rollout policy that produced the sampled trajectory. We assign a constant negative reward at each non-terminal decoding step:

rt = −(1 − γ), t = 0, . . . , L − 1, (1)

where γ ∈ (0,1) is the discount factor, and define rL = 0. The factor (1 − γ) is introduced purely for normalization, so the resulting return lies in a fixed range compatible with a sigmoid-bounded output.

The discounted return from step t is

L−t

γirt+i = −(1 − γL−t). (2)

∑

Gt ≜

i=0

Thus, for each non-terminal step, Gt ∈ (−1,0) and is a strictly monotone function of the remaining generation length L − t: states closer to termination have values nearer to 0, while states with longer continuations have values closer to −1. Compared with raw length, this bounded transform preserves ordering while improving target scale and conditioning. It also satisfies the Bellman recursion

Gt = rt + γGt+1, (3) placing token-level length modeling in a standard value-learning framework.

Key properties of the target. The return construction has three useful properties. First, it is bounded: Gt ∈ (−1,0) for all non-terminal steps, avoiding the large dynamic range of raw remaining length. Second, it is monotone: the ordering over remaining horizons is preserved exactly, so shorter continuations always map to values closer to 0. Third, it is Bellman-consistent: the target obeys a one-step recursion, which places token-level length modeling in the same formal framework as standard discounted value estimation.

The discount factor γ controls how aggressively long horizons are compressed. Larger γ preserves more resolution over long continuations, while smaller γ concentrates resolution near termination. Equivalently, γ determines the logarithmic scale on which remaining length is represented.

For a fixed model-induced decoding policy π, the corresponding length value function is

Vπ(st) = Eπ[Gt ∣ st]. (4) In practice, we sample trajectories from a fixed rollout policy and regress to their realized returns as Monte Carlo supervision. When applied to a realized return, the transform can be inverted exactly to recover the realized remaining token count:

loglog(1 +γGt)

. (5)

l =

When applied to a predicted conditional mean value, however, this inverse yields a proxy-derived length estimate rather than, in general, the conditional expectation of raw remaining length. We discuss this distinction in Appendix F.

This formulation is (1) annotation-free, since targets are computed directly from observed completion lengths; (2) dense, since every non-terminal step contributes a regression target; (3) unbiased for the policy-conditioned value target, since under a fixed rollout policy the realized return is an unbiased Monte Carlo sample of Vπ(st) and is computed deterministically from the sampled completion, introducing no additional annotation or reward-model noise; and (4) scalable, since multiple completions can be sampled per prompt to automatically construct many supervised trajectories.

###### 3.2. LenVM Head

LenVM is built on top of LLMs or VLMs by attaching a scalar value head to the final-layer hidden state at each decoding step:

zt = head(ht). (6) In this work, we instantiate head(⋅) as a two-layer MLP with SiLU activation,

zt = W2 SiLU(W1ht + b1) + b2, (7) followed by

Vθ(st) = −σ(zt), (8) which ensures Vθ(st) ∈ (−1,0).

###### 3.3. Training Objective

For a minibatch of N prompt-completion trajectories, where the n-th trajectory has generated length L(n), we optimize the token-averaged mean squared error

t=0 (Vθ(s(tn)) − Gt(n))2 ∑nN=1 L(n)

(n)−1

∑nN=1 ∑L

, (9)

ℒlen =

Table 1. Datasets used to train general LenVMs.

Domain Dataset Scale Code OpenCodeReasoning-2 (Ahmad et al., 2025) (Python) 1.42M Instruction Following WildChat (Zhao et al., 2024) 529k Math DeepMath-103K (He et al., 2025) 103k

where

Gt(n) = −(1 − γL

(n)−t). (10)

Here the trajectories are sampled from a fixed generator checkpoint and decoding policy, and Gt(n) is computed exactly from each realized completion. This objective therefore corresponds to Monte Carlo regression with dense token-level supervision over the rollout state distribution induced by that policy.

More generally, one may consider weighted variants of the same objective, where each token-level squared error is multiplied by a weight. In our setting, we avoid weights that depend on future rollout outcomes relative to the current decoding step, since such weights change the effective regression objective. Trajectorylevel averaging (token-mean then sequence-mean) is one such case, as it induces a per-token weight inversely proportional to the realized completion length. We discuss this issue in Appendix E.

Relation to GAE. LenVM also admits GAE (Schulman et al., 2015). With δt = rt + γVθ(st+1) − Vθ(st), one can construct bootstrapped targets from temporal-difference residuals. This provides an optional bootstrapped variant of the same value-learning problem. In our experiments, however, λ = 1 performs best. This is consistent with the task structure: supervision is dense and directly recoverable from completed sequences, so full-return regression is already available without additional bootstrapping assumptions.

#### 4. Experiments: Validating LenVM as a Token-Level Length Signal

In this section, we evaluate at inference time how well LenVM captures the token-level signal of remaining generation length. Our experiments ask three questions: (i) Can LenVM support inference-time control without modifying the base generator? (ii) Can it predict generation length from the prompt boundary? (iii) Does its value-learning objective scale with model and data? We answer these questions through four evaluations: (1) length-controlled generation (§4.2), (2) performance–efficiency trade-off (§4.3), (3) generation length prediction (§4.4), and (4) scalability of LenVM (§4.5).

We note that applying LenVM during decoding requires additional forward passes and therefore introduces extra inference latency. The end-to-end latency is also affected by many engineering factors. Accordingly, our experiments are intended primarily to validate LenVM’s ability to model and control generation length, rather than to optimize wall-clock latency.

###### 4.1. Experimental Setup

General LenVM. We train LenVMs on a multi-domain mixture of math, code, and instruction-following data rather than training separate domain-specific models. Table 1 summarizes the training mixture. For each prompt, we sample multiple completions and convert them into dense per-token regression targets using the discounted return in §3.

Models. For Qwen2.5 experiments (both LLMs and VLMs), we initialize LenVM from the Qwen2.5-Instruct family (Yang et al., 2024). For Qwen3 experiments, we initialize LenVM from Qwen3-Base models (Yang et al., 2025).

More experimental detail are elaborated in Appendix A.

###### 4.2. Application 1: Length-Controlled Generation

We first evaluate whether LenVM can provide inference-time control signals for explicit length constraints. We evaluate on LIFEBench (Zhang et al., 2025b), which contains 360 instances across question answering, summarization, reasoning, and creative generation, with 180 Chinese and 180 English examples. LIFEBench defines three settings: Equal To, At Most, and At Least. Following LIFEBench, we report Length Score (LS↑) for all settings and Length Deviation (LD↓) for Equal To. We consider target lengths from 32 to 1024 tokens. More details of LIFEBench and our modification are given in Appendix B.

Hard-constraint decoding with LenVM. At each step, we first restrict the vocabulary to a candidate set 𝒱t using standard truncation strategies, then score each candidate token with LenVM on the next state vˆ(x) = vˆϕ(st ⊕ x). We then select tokens using:

- • Equal To: First convert the remaining target length L − t at into value space v∗t and select argminx∈𝒱t ∣vˆ(x) − v⋆t ∣, favoring candidates whose predicted value is closest to the target v⋆t .
- • At Least: argminx∈𝒱t vˆ x , favoring more negative values and hence longer continuations.
- • At Most: argmaxx∈𝒱t vˆ(x), favoring values closer to 0 and hence earlier termination.

Results. Table 2 shows that LenVM substantially improves constraint following, especially in the Equal To setting. For example, on Qwen2.5-7B-Instruct, LenVM improves Length Score from 30.9 to 64.8 and reduces deviation from 71% to 44%. The comparison to closed-source models is also informative: those models rely on prompt-based control, which is inherently coarse-grained, and they struggle on Equal To, with deviations between 66% and 135% and scores below 54. By contrast, LenVM provides an explicit token-level control signal over the remaining horizon, allowing open models to achieve substantially stronger precision.

###### 4.3. Performance–Efficiency Trade-off

We next test whether LenVM supports continuous control of the performance–efficiency trade-off without modifying the base generator. Instead of enforcing a hard target length, we reweight the next-token distribution so that tokens predicted to lead to shorter continuations become more likely.

Value-guided exponential tilting. Let p be the base model distribution over candidate set 𝒱t. We define

DKL(p′∥p), (11) whose solution is

Ep′[vˆ(x)] −

1 β

min

p′

p(x)exp(βvˆ(x)) ∑x′∈𝒱t p(x′)exp(βvˆ(x′))

p′(x) =

. (12)

- Table 2. Results on LIFEBench. We compare LenVM-guided hard-constraint decoding against frontier closed-source models. Following the LIFEBench reporting format, we present Equal To (Length Deviation↓, Length Score↑), At Most (Length Score↑), and At Least (Length Score↑). We report target lengths from 32 to 1024 tokens. LenVM equips open models to significantly outperform frontier proprietary models in exact length matching.

Equal To At Most At Least

Model

Deviation↓ Score↑ Score↑ Score↑

Closed-source frontier models GPT-4o 74% 35.5 77.9 98.5 GPT-5.4 135% 37.4 65.4 98.9 GPT-5.4-thinking 131% 47.8 72.7 98.9 Claude-Sonnet-4-6 105% 34.1 62.9 100.0 Claude-Sonnet-4-6-thinking 124% 51.3 69.3 100.0 Claude-Opus-4-6 66% 35.5 51.5 100.0 Claude-Opus-4-6-thinking 87% 53.2 67.4 100.0 Gemini-3-Flash-Preview 123% 40.3 57.3 99.6 Gemini-3.1-Pro-Preview 91% 49.3 70.7 100.0

Qwen2.5-3B-Instruct

w/o LenVM 83% 25.6 92.1 94.6 w/ LenVM(1.5B) 56% (↓27pp) 62.6 (↑37.0) 93.0 93.1

Qwen2.5-7B-Instruct

w/o LenVM 71% 30.9 98.5 89.1 w/ LenVM(1.5B) 44% (↓27pp) 64.8 (↑33.9) 96.1 99.5

Qwen3-30B-A3B-Instruct

w/o LenVM 90% 36.8 87.0 99.3 w/ LenVM(1.7B) 57% (↓33pp) 67.2 (↑30.4) 99.4 99.8

The derivation is provided in Appendix G. Here β < 0 controls the steering strength. When β = 0, decoding reduces to the original model. As β becomes more negative, generation is increasingly biased toward shorter trajectories.

We evaluate on GSM8K (Cobbe et al., 2021), MATH500 (Lightman et al., 2023), and MathVista (Lu et al., 2023). As a baseline, we use a hard token budget that truncates generation once the token count exceeds threshold B and marks the example as incorrect. For fair comparison, the x-axis reports the average truncated generation length for both methods.

Figure 2 shows that LenVM-guided decoding consistently achieves a better performance–efficiency frontier than hard truncation. At the same average truncated length, LenVM retains much higher Pass@1. For example, on GSM8K with Qwen2.5-3B-Instruct, at roughly 200 tokens the hard budget baseline achieves about 6% Pass@1, whereas LenVM maintains about 63%. Because the base generator is never modified, this result indicates that the model itself already contains shorter successful reasoning paths. LenVM simply biases decoding toward them. Moreover, varying β changes the frontier smoothly, confirming that LenVM provides a continuous knob for balancing performance and token budget.

=-50 =-20 =0

=0

=-50 =-20 =0

=-20

=-100

=-50

0.275

=-100

0.8

| |
|---|

=-100

| |
|---|

| |
|---|

B=

B=

B=

| |
|---|

| |
|---|

0.5

| |
|---|

0.250

=-1000

B=400

0.7

B=450

B=800

=-1000

| |
|---|

0.225

=-1000

B=300

B=400

0.6

0.4

B=600

0.200

Pass@1

Pass@1

Pass@1

0.5

B=350

0.175

B=500

0.4

0.3

B=200

0.150

B=300

0.3

B=400

0.125

0.2

0.2

Token Budget (Baseline)

Token Budget (Baseline)

Token Budget (Baseline)

0.100

B=250

0.1

LenVM (Ours)

LenVM (Ours)

LenVM (Ours)

0.075

0.1

B=200

B=300

B=150

200 225 250 275 300 325

300 350 400 450 500 550

140 160 180 200 220 240

Avg Response Length (tokens)

Avg Response Length (tokens)

Avg Response Length (tokens)

(a) GSM8K; Base (Qwen2.5-3B-Instruct)

(b) MATH500; Base (Qwen2.5-7B-Instruct)

(c) MathVista; Base (Qwen2.5-VL-7B-Instruct)

+ LenVM (Qwen2.5-VL-3B-Instruct)

+ LenVM (Qwen2.5-1.5B-Instruct)

+ LenVM (Qwen2.5-1.5B-Instruct)

Figure 2. Performance (Pass@1) versus average generation length. LenVM-guided exponential tilting (red) consistently outperforms a hard token-budget baseline (blue). The x-axis reports the average truncated response length. Since the base generator is unchanged, the result shows that the model already has the ability to solve problems with shorter generations, and LenVM helps uncover those trajectories by reweighting token choices. Models in parentheses represent the initialization checkpoints.

###### 4.4. Application 3: Generation Length Prediction

Beyond control, we test whether LenVM can predict generation length from the prompt boundary, i.e., from state s0 before any response token is emitted. This is useful for scheduling, batching, and memory planning at inference time.

To evaluate prediction accuracy, we use held-out prompts from math, code, and instruction-following data. For each prompt, we sample N = 64 completions with lengths L1, . . . , LN. Because LenVM predicts discounted remaining horizon rather than raw token count, we evaluate in the same transformed space:

N

u(Li), (13)

u(Li) = 1 − γLi, µu =

1 N

∑

i=1

and define the return-consistent ground-truth horizon as

ln(1ln−γµu)

. (14)

LGT =

We then invert LenVM’s prediction at s0 into a length estimate Lˆ and report Mean Relative Error (MRE):

∣Lˆ −LGTLGT∣]. (15) Further discussion of this target appears in Appendix F.

MRE(Lˆ, LGT) ≜ E[

- As shown in Table 3, LenVM predicts generation horizon from the prompt boundary with low relative error across all domains, and accuracy improves consistently with scale. At 32B, MRE reaches 9.8% on math, 14.9% on code, and 17.1% on instruction following. This shows that LenVM is not only useful for online control, but also captures predictive information about expected generation length before decoding begins.

- Table 3. Prompt-boundary length prediction accuracy. Mean Relative Error (MRE, lower is better) on three domains.

MRE (↓)

Model Size

Math Code IF

1.5B 17.0% 29.0% 33.0% 3B 13.6% 24.0% 27.2% 7B 11.0% 19.5% 23.0%

14B 10.4% 17.0% 19.8% 32B 9.8% 14.9% 17.1%

0.0085 Num of Questions

0.009

Samples

Model Size

0.008

- n=1

- n=2

- 0.5b

- 1.5b

10k 30k 100k

0.0080

0.008

3b 7b

n=4

0.0075

0.007

TestLoss

n=16

0.007

14b 32b

0.0070

0.006

0.006

0.0065

0.0060

0.005

0.005

0.0055

0 500 1000 1500

0 250 500 750 1000

0 1000 2000

Step

Step

Step

(a) Model size

(b) Number of training questions

(c) Samples per question

Figure 3. LenVM scales along three axes. Validation loss decreases consistently as we increase (a) model size, (b) number of training questions, and (c) samples per question.

###### 4.5. Scalability of LenVM

A key advantage of LenVM is that its supervision is annotation-free, dense, and easy to scale. We therefore study scaling along three axes: (i) model size, (ii) number of training questions, and (iii) number of samples per question.

- Figure 3 shows consistent improvements along all three axes. Larger models achieve lower validation loss, matching the gains in prompt-boundary prediction accuracy from Table 3. Increasing the number of training prompts also improves performance, indicating that LenVM benefits from broader coverage of tasks and trajectory types. Finally, increasing the number of sampled completions per prompt further reduces validation loss, showing that supervision can be scaled not only through more prompts but also through more sampled trajectories from the same prompts. Overall, these results support LenVM as a scalable value-pretraining objective.

###### 4.6. Case Study: Length Tokens as Markers of Length Shifts

Beyond prediction and control, LenVM also offers a qualitative view of where generation shifts toward longer or shorter continuations. We analyze tokens that repeatedly co-occur with upward or downward changes in LenVM’s predicted remaining horizon, and refer to them as length tokens. For each decoding step, we record LenVM’s value prediction Vt ≜ Vθ(st). We then measure the local change in predicted remaining horizon

[Figure 12]

[Figure 13]

(a) Positive length tokens: longer-horizon shifts (b) Negative length tokens: shorter-horizon shifts

- Figure 4. Length tokens as markers of horizon shifts. Word clouds show tokens whose one-step TD residual st (Eq. 16) frequently exceeds a positive or negative threshold 0.01. We use the completion from Qwen3-30B-A3Binstruct and 8k data from DeepMath-103K dataset. Token size is proportional to frequency. “emoji_check” denotes the check-mark emoji and “emoji_party” denotes the party emoji.

across consecutive prefixes using the one-step temporal-difference (TD) residual st ≜ rt−1 + γVt − Vt−1, rt−1 = −(1 − γ), (16)

where γ is the same discount factor used in training. Equivalently, st is a TD-style score that captures whether the newly extended prefix leads LenVM to revise its estimate of the remaining horizon upward or downward.

Large positive st indicates a shift toward a longer expected continuation, while large negative st indicates a shift toward a shorter one. We count the tokens that st are greater than or less than 0.01 and plot a word cloud according to their frequency of occurrence. We use the completion from Qwen3-30B-A3B-instruct and 8k data from DeepMath-103K dataset.

Figure 4a shows that positive length tokens often resemble local reasoning pivots, including ah, but, now, wait, let, think, try, and consider. These tokens often appear when the model revises an earlier step, introduces a new subcase, or extends the current line of reasoning. In particular, ah frequently occurs in contexts such as “Ah! I see the mistake” or “Ah! Here’s an idea,” which are reminiscent of Aha Moment where the model has an epiphany.

By contrast, in Figure 4b, negative length tokens such as therefore, clearly, perfect, \n\n, and celebratory emoji markers such as check-mark emoji or party emoji are more often associated with closure, confirmation, or answer finalization. This analysis is descriptive rather than causal, but it suggests that LenVM provides not only a control signal, but also a simple token-level lens on generation dynamics.

- 5. Ablations

We ablate four design choices in LenVM: (i) length-space representation, (ii) batch construction strategy, (iii) discount factor γ, and (iv) numerical precision (fp16/bf16/fp32). Unless otherwise specified, all experiments keep the model, optimizer, and evaluation protocol fixed.

- 5.1. Length-Space Representation

We first compare four target parameterizations in Figure 5a: (1) Length + Softplus, (2) Normalized Length + Sigmoid, (3) Log Length + Softplus, and our proposed (4) Discount Return + Sigmoid. Among them,

AverageAbsoluteLengthLoss

Setting

Setting

0.016

0.008

250

Setting

bf16(default)

no shuffle

0.014

Length + Softplus

fp16 fp32

shuffle

Normed Length + Sigmoid

TestLoss

200

TestLoss

0.012

0.007

Log Length + Softplus

Discount Return

0.010

150

0.006

0.008

100

0.006

0.005

0.004

0 250 500 750 1000 Step

0 250 500 750 1000 Step

0 250 500 750 1000

Step

(a) Length-space representation

(b) Shuffling vs. grouped batching

(c) Numerical precision

- Figure 5. Ablations on LenVM design choices. Comparison of (a) target representation, (b) batch construction strategy, and (c) numerical precision.

0 500 1000 1500

Step

0.225

0.250

0.275

0.300

0.325

LengthRelativeError

Gamma

0.995 0.999 0.99

(a) 0%

0 500 1000 1500

Step

0.25

0.30

0.35

Gamma

0.995 0.999 0.99

(b) 25%

0 500 1000 1500

Step

0.275

0.300

0.325

0.350

Gamma

0.995 0.999 0.99

(c) 50%

0 500 1000 1500

Step

0.30

0.35

0.40

0.45 Gamma

0.995 0.999 0.99

(d) 75%

- Figure 6. Ablation on the discount factor γ. Prediction error at different relative decoding positions. Larger γ performs better earlier in generation, while smaller γ performs better closer to termination.

Discount Return + Sigmoid consistently achieves the lowest average absolute length error throughout training. Raw-length and normalized-length regression perform worse, partly because generation lengths are highly long-tailed. In particular, normalizing by the maximum length compresses the dense short-length region into a narrow interval near zero, where discrimination becomes difficult. Log-length regression is a stronger baseline and substantially improves over raw or normalized length, but still underperforms discount return. Unlike a static scale transform such as log length, discount return is aligned with the stepwise Bellman recursion of autoregressive decoding, yielding a more coherent token-level training signal. This result shows that for token-level length modeling, directly regressing raw length or normalized length is ineffective, while the discounted-return formulation provides a better-conditioned and more effective target.

###### 5.2. Batch Construction Strategy

We compare fully shuffled training against grouped batching that keeps multiple completions from the same prompt together, while holding the total number of data per update fixed. As shown in Figure 5b, shuffling consistently improves evaluation loss. This suggests that, unlike the reward model, grouping multiple samples from the same prompt within a batch is unnecessary and can mildly hurt generalization.

###### 5.3. Discount Factor γ

The discount factor γ determines how remaining horizon is distributed in the target space, and therefore where prediction resolution is concentrated along the trajectory. Larger γ compresses long horizons more aggressively, which tends to improve prediction earlier in generation. Smaller γ allocates more resolution near termination, improving prediction later in the trajectory. To evaluate this trade-off, we measure prediction error at relative decoding positions 0%, 25%, 50%, and 75%. Figure 6 shows the expected pattern: larger γ performs better at earlier positions, while smaller γ performs better closer to the end. Thus, γ acts as a simple knob for balancing early-horizon compression against late-horizon discrimination. In practice, we choose an intermediate γ to balance early-horizon compression against late-horizon discrimination. Specifically, we set γ such that the 99th-percentile generation length L0.99 satisfies 1 − γL0.99 = 0.99, so that nearly all observed horizons are mapped into the high-resolution region of the target space.

###### 5.4. Numerical Precision

We compare fp16, bf16, and fp32 under the same training and evaluation setup. Figure 5c shows nearly identical loss curves across all three formats, with no substantial difference in convergence or final performance. This indicates that LenVM is numerically stable under common floating-point precisions. Appendix D provides further analysis of finite-precision effects.

#### 6. Conclusion

We proposed LenVM, a token-level value model for remaining generation length. By assigning a constant per-token reward and predicting the resulting discounted return, LenVM places length modeling in a standard value-learning framework with bounded, dense, and annotation-free supervision. Our experiments show that this simple formulation is both useful and scalable. LenVM supports precise length-controlled generation, exposes a smooth performance–efficiency trade-off without modifying the base generator, predicts expected generation horizon, and improves consistently with model scale and automatically collected supervision. These results suggest that discounted length value is an effective target for token-level length modeling and a promising substrate for scalable value pretraining.

#### References

Wasi Uddin Ahmad, Somshubra Majumdar, Aleksander Ficek, Sean Narenthiran, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning-ii: A simple test time scaling approach via self-critique. arXiv preprint arXiv:2507.09075, 2025.

Adewale Akinfaderin, Shreyas Subramanian, and Akarsha Sehwag. Plan-and-write: Structure-guided length control for llms without model retraining. ArXiv, abs/2511.01807, 2025. URL https://api. semanticscholar.org/CorpusID:282739780.

Bradley Butcher, Michael O’Keefe, and James Titchener. Precise length control for large language models. Nat. Lang. Process. J., 11:100143, 2024. URL https://api.semanticscholar.org/CorpusID: 274788732.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yifeng Ding, Hantian Ding, Shiqi Wang, Qing Sun, Varun Kumar, and Zijian Wang. Planning-aware code

infilling via horizon-length prediction, 2025. URL https://arxiv.org/abs/2410.03103.

Emmanuel Anaya Gonzalez, Sairam Vaidya, Kanghee Park, Ruyi Ji, Taylor Berg-Kirkpatrick, and Loris D’antoni. Constrained sampling for language models should be easy: An mcmc perspective. ArXiv, abs/2506.05754,

##### 2025. URL https://api.semanticscholar.org/CorpusID:279245064.

Yuxuan Gu, Wenjie Wang, Xiaocheng Feng, Weihong Zhong, Kun Zhu, Lei Huang, Tat-Seng Chua, and Bing Qin. Length controlled generation for black-box llms. ArXiv, abs/2412.14656, 2024. URL https: //api.semanticscholar.org/CorpusID:274859461.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Chenghua Huang, Lu Wang, Fangkai Yang, Pu Zhao, Zhixu Li, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Pretrain value, not reward: Decoupled value policy optimization, 2026. URL https://arxiv.org/abs/2502.16944.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. Proceedings of the 29th Symposium on Operating Systems Principles, 2023. URL https: //api.semanticscholar.org/CorpusID:261697361.

Yanhao Li, Lu Ma, Jiaran Zhang, Lexiang Tang, Wentao Zhang, and Guibo Luo. Leash: Adaptive length penalty and reward shaping for efficient large reasoning model, 2025. URL https://arxiv.org/abs/ 2512.21540.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Dler: Doing length penalty right incentivizing more intelligence per token via reinforcement learning, 2025. URL https://arxiv.org/ abs/2510.15110.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Ning Miao, Hao Zhou, Lili Mou, Rui Yan, and Lei Li. Cgmh: Constrained sentence generation by metropolis-

##### hastings sampling, 2018. URL https://arxiv.org/abs/1811.10996.

Grzegorz Piotrowski, Mateusz Bystroński, Mikołaj Hołysz, Jakub Binkowski, Grzegorz Chodak, and Tomasz Jan Kajdanowicz. When will the tokens end? graph-based forecasting for LLMs output length. In Jin Zhao, Mingyang Wang, and Zhu Liu, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 843–848, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-254-1. doi: 10.18653/v1/2025. acl-srw.61. URL https://aclanthology.org/2025.acl-srw.61/.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. ArXiv, abs/2211.05102, 2022. URL https://api.semanticscholar.org/CorpusID:253420623.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. CoRR, abs/1506.02438, 2015. URL https://api. semanticscholar.org/CorpusID:3075448.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017. URL https://api.semanticscholar.org/CorpusID: 28695052.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. ArXiv, abs/2408.03314, 2024. URL https: //api.semanticscholar.org/CorpusID:271719990.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Feng Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Haochen Ding, HaoXing Hu, Haoming Yang, Hao Zhang, Haotian Yao, Hao-Dong Zhao, Haoyu Lu, Haoze Li, Hao Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jia-Xing Guo, Jianling Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Li-Na Shi, Li tao Ye, Long Yu, Meng xiao Dong, Neo Y. Zhang, Ning Ma, Qi Pan, Qucheng Gong, Shaowei Liu, Shen Ma, Shu-Yan Wei, Sihan Cao, Si-Da Huang, Tao Jiang, Wei-Wei Gao, Weiming Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wen He, Xian sen Wei, Xian-Xian Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xue biao Pan, Y. Charles, Yang Li, Yan-Ling Hu, Yangyang Liu, Yanru Chen, Ye-Jia Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Yingbo Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhengxin Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziya Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. ArXiv, abs/2501.12599, 2025. URL https://api.semanticscholar.org/CorpusID:275789974.

Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, and Nick Haber. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning, 2025. URL https://arxiv.org/abs/2506.05256.

Meiman Xiao, Ante Wang, Qingguo Hu, Zhongjian Miao, Huangjun Shen, Longyue Wang, Weihua Luo, and Jinsong Su. Can llms track their output length? a dynamic feedback mechanism for precise length regulation. ArXiv, abs/2601.01768, 2026a. URL https://api.semanticscholar.org/CorpusID:284486970.

Meiman Xiao, Ante Wang, Qingguo Hu, Zhongjian Miao, Huangjun Shen, Longyue Wang, Weihua Luo, and Jinsong Su. Can llms track their output length? a dynamic feedback mechanism for precise length regulation, 2026b. URL https://arxiv.org/abs/2601.01768.

Huanyi Xie, Yubin Chen, Liangyu Wang, Lijie Hu, and Di Wang. Predicting llm output length via entropyguided representations. ArXiv, abs/2602.11812, 2026. URL https://api.semanticscholar.org/ CorpusID:285540500.

Juncheng Xie and Hung yi Lee. Prompt-based one-shot exact length-controlled generation with llms. ArXiv,

##### abs/2508.13805, 2025. URL https://api.semanticscholar.org/CorpusID:280686321.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Jingren Zhou, Junyan Lin, Kai Dang, Keqin Bao, Ke-Pei Yang, Le Yu, Li-Chun Deng, Mei Li, Min Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shi-Qiang Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. 2025. URL https://api.semanticscholar.org/CorpusID: 278602855.

Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan, and Zekun Wang. Qwen2.5 technical report. ArXiv, abs/2412.15115, 2024. URL https://api.semanticscholar.org/CorpusID:274859421.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks, 2025. URL https://arxiv.org/abs/2504.05118.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. ArXiv, abs/2505.13417, 2025a. URL https://api.semanticscholar.org/CorpusID: 278769267.

Wei Zhang, Zhenhong Zhou, Kun Wang, Junfeng Fang, Yuanhe Zhang, Rui Wang, Ge Zhang, Xavier Li, Li Sun, Lingjuan Lyu, et al. Lifebench: Evaluating length instruction following in large language models. arXiv preprint arXiv:2505.16234, 2025b.

Yi-Kai Zhang, Zhiyuan Yao, Hongyan Hao, Yueqing Sun, Qi Gu, Hui Su, Xunliang Cai, De-Chuan Zhan, and

Han-Jia Ye. v0: A generalist value model for any policy at state zero, 2026. URL https://arxiv.org/ abs/2602.03584.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatgpt interaction logs in the wild. arXiv preprint arXiv:2405.01470, 2024.

Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, and Yang You. Response length perception and sequence scheduling: An llm-empowered llm inference pipeline, 2023. URL https://arxiv.org/ abs/2305.13144.

#### Appendix

#### A. Additional Experimental Details

We train LenVM using LlamaFactory. The discount factor γ is set to 0.997 for Qwen2.5-Instruct and Qwen2.5-VL-Instruct, and to 0.9998 for Qwen3-Instruct. We set λ = 1 for GAE.

For the ablation and scalability experiments, we sample 100k examples from each dataset, except for math, where we use all 95k available examples. For the Length-Controlled Generation and Performance–Efficiency Trade-off experiments, we use the full training sets of all three datasets. We randomly sample 8k examples for validation during training. Unless otherwise specified, data sampling uses temperature 1.0 and top-p 1.0, and we sample up to 16 completions per prompt.

We train LenVM for 2 epochs with learning rate 2 × 10−5, batch size 1024, and BF16 precision.

In Length-Controlled Generation, we use temperature 1.0, top-p 0.999, and top-k 15. All baseline settings follow the original LIFEBench paper, except that constraints are applied to token count in our experiments. We use the official tokenizer of each model, or the token count returned by the corresponding API response, to measure generation length. Although tokenization schemes differ across models, all methods are evaluated under the same token-based constraint setting.

In the Performance–Efficiency Trade-off experiments, we use temperature 1.0 and top-p 1.0. For LenVM guided decoding, we set min-p to 0.01 to reduce the candidate set size and avoid excessively large LenVM forward batches. We sample 64 completions for each question.

#### B. LIFEBench Evaluation Details

Benchmark overview. LIFEBench (Zhang et al., 2025b) is a comprehensive benchmark designed to evaluate the ability of large language models to follow explicit length instructions across diverse tasks and length scales. It comprises 10,800 (360 data, 3 length constraints, 10 length target) instances spanning four task categories:

- • Question Answering (QA): Open-ended questions drawn from Quora, Stack Exchange, WikiHow, and Yahoo Answers (English), and from Zhihu and WikiHow (Chinese).
- • Summarization: Long documents (10,000–15,000 words) sourced from scientific papers, government reports, book chapters, and Wikipedia (English), and from corporate annual reports, Wikipedia, and XueXiQiangGuo (Chinese).
- • Reasoning: Eighty problems (40 per language) generated via GPT-4o, covering six reasoning types: deductive, inductive, abductive, analogical, causal, and probabilistic.
- • Creative Generation: Writing prompts derived from movie scripts, story corpora, and news datasets (English), and from web novels, social commentary, and WeChat public articles (Chinese).

Length constraints. The original benchmark focuses on word and character lengths. Each instance is paired with one of three constraint types (Equal To, At Most, and At Least) and one of ten target lengths: 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, and 8,192 words for English or characters for Chinese. In our experiments, we modify these constraints to operate at the token level. Specifically, we prompt the model to generate a response where the total token count is equal to, at most, or at least the specified target length.

Evaluation metrics. Two metrics are used:

###### • Length Deviation (LD):

Loutput − Lconstraint Lconstraint

, (17)

LD =

where Loutput is the actual output length and Lconstraint is the target length. Positive values indicate over-generation; negative values indicate under-generation.

###### • Length Score (LS): A bounded score in [0,100] derived from LD via an exponential penalty function that is adapted to each constraint type:

⎧ ⎨ ⎪⎩

100 ⋅ ek1⋅LD, LD < 0 Equal To / At Least), 100 ⋅ e−k2⋅LD, LD > 0 (Equal To / At Most), 100, otherwise,

(18)

LS =

with k1 = 5 and k2 = 2. A score of 100 indicates perfect adherence; lower scores reflect larger deviations.

Implementation details. Each prompt follows the template [Instruction] + [Requirement], where the instruction specifies the task and input content, and the requirement states the length constraint in natural language (e.g., “Your answer must be equal to 512 tokens.”).

#### C. LenVM as a Length-Specific Value Function in RL

This section makes explicit the RL interpretation of LenVM and clarifies two distinct ways in which it can interact with reinforcement learning. First, the return constructed in Section 3 is already a valid discounted reward process, so LenVM can be viewed as a value function for a token-level length objective. Second, because LenVM defines a state-dependent notion of progress toward termination, it can also be used to construct potential-based shaping rewards that provide dense intermediate learning signals without changing the underlying task objective, under the standard fixed-potential assumptions used in reward shaping. Our use of this connection in the main paper is conceptual only: we do not perform RL fine-tuning with LenVM, and all reported results use LenVM at inference time.

###### C.1. LenVM as a Value Function for a Length Objective

Consider an autoregressive policy π over a generated trajectory of length L. Suppose the RL objective contains both a task component and a length component,

J(π) = Jtask(π) + s Jlen(π), (19)

where s ∈ R controls the strength and sign of the length term. Following Section 3.1, define the per-step length reward as

rtlen = −(1 − γ), t = 0, . . . , L − 1, (20) with terminal reward rlenL = 0. The corresponding discounted return from state st is then

L−t

γirtlen+i = −(1 − γL−t). (21)

∑

Gtlen =

i=0

Therefore the length value function under policy π is Vπlen(st) ≜ Eπ[Gtlen ∣ st]. (22)

This is exactly the quantity LenVM is trained to estimate. In this sense, LenVM is not merely correlated with generation length. It is a value function for a well-defined token-level objective in which each additional decoding step incurs a constant discounted cost.

This interpretation is useful because it makes explicit what happens if LenVM is inserted directly into RL optimization. Using a positive coefficient s in the additive objective above means that the policy is optimized

notgenerallyonly forchangestask reward,the taskbutobjectivealso for shorterunless generationcontinuations.efficiencyThus, directis itselfoptimizationpart of theofintendedJtask(π)goal.+ sJlenThis(π) distinction is important. LenVM can be used either as a genuine objective component, when one wishes to trade off quality against generation cost, or as a source of auxiliary structure for improving learning without redefining the task. The latter case motivates the shaping perspective developed below.

###### C.2. Advantage Decomposition and Policy Optimization

With separate task and length rewards, one can also maintain separate value baselines. Let Vtask(st) denote the value function for the task reward and let vˆϕ(st) denote the LenVM prediction for the length reward. Then the temporal-difference residuals can be written as

δttask = rttask + γtaskVtask(st+1) − Vtask(st), (23) and

δtlen = rtlen + γvˆϕ(st+1) − vˆϕ(st). (24) Applying GAE to each component gives

L−1−t

(γtaskλtask)iδttask+i , (25)

∑

Ataskt =

i=0

and

L−1−t

(γλlen)iδtlen+i. (26) In the simplest case, one combines these signals additively:

∑

Alent =

i=0

Atotalt = Ataskt + s Alent . (27)

One may also schedule the scalar coefficient s over training if the intended quality-efficiency trade-off changes over time. Here s > 0 discourages longer continuations, while s < 0 encourages them.

This decomposition is attractive because the quality and efficiency terms remain explicitly separated. The task critic models the utility of the generated content, while LenVM models the expected future cost of continuing generation. As a result, changing the scalar coefficient changes how these two factors enter the policy update without redefining the underlying value targets themselves.

- At the same time, it is important to note that this additive formulation is an objective change, not merely a variance-reduction device. When s ≠ 0, the induced policy update explicitly prefers trajectories with different length characteristics. Therefore this formulation is most appropriate when one truly wishes to optimize a performance-efficiency trade-off, such as token-budgeted generation, latency-aware decoding, or resource-constrained RL for language models.

###### C.3. LenVM as a Potential for Policy-Invariant Reward Shaping

The same learned signal also admits a different use that does not directly modify the underlying task objective. Suppose the original RL problem is defined only by task reward rttask. If one wishes to provide dense intermediate feedback without changing the underlying task objective, a standard construction is potential-based reward shaping:

rt′ = rttask + β(γΦ(st+1) − Φ(st)), (28) where Φ is any scalar potential over states and β ≥ 0 controls the shaping strength.

LenVM provides a natural candidate for such a potential. Because vˆϕ(st) ∈ (−1,0) is a bounded monotone transform of expected remaining generation length, it defines a notion of prefix-wise progress toward termination. States closer to EOS have values nearer to 0, whereas states expected to continue longer have values closer to −1. A simple choice is therefore

Φ(st) = sg(vˆϕ(st)), (29) where sg(⋅) denotes stop-gradient, and we set

Φ(sEOS) = 0. (30) The resulting shaping term is

Ft = γΦ(st+1) − Φ(st). (31)

This construction clarifies the role of LenVM in RL. If Φ(st+1) > Φ(st), then the next decoding state is predicted to be closer to termination, so Ft becomes larger. Such a transition receives a more positive shaping signal. Conversely, if Φ(st+1) < Φ(st), then the chosen token moves the trajectory toward a longer continuation, and the shaping signal becomes smaller or more negative. Thus LenVM-based shaping does not directly reward short outputs as an endpoint property. Rather, it rewards local transitions that appear to make progress toward termination under the current decoding dynamics.

Under the standard shaping assumptions that the decoding process is treated as a discounted Markov decision process, the same discount factor is used in both the base return and the shaping term, and the potential is fixed during policy optimization, the reason this shaping can alter optimization dynamics without changing the underlying task objective is the telescoping form of the added reward. For any trajectory starting from s0,

T

γt(γΦ(st+1) − Φ(st)) = −Φ(s0) + γT+1Φ(sT+1). (32)

∑

t=0

When terminal states are assigned zero potential, this reduces to a boundary term depending only on the start state. Therefore all trajectories from the same initial state receive the same offset, and their ordering under expected return is unchanged. In this sense, potential-based shaping modifies the distribution of intermediate learning signals rather than the final task criterion itself.

This distinction is especially relevant for language-model RL. A direct length penalty changes what the policy is asked to optimize. By contrast, LenVM-based shaping can be interpreted as a progress signal that redistributes terminal information over token-level decisions. It may therefore improve credit assignment and sample efficiency even when the intended objective is still purely task success.

In practice, however, LenVM is policy-dependent, since it estimates Vπlen(st) under the rollout policy. For the shaping interpretation to remain closest to the classical policy-invariance result, it is preferable to use a

pretrained and fixed LenVM when constructing Φ. If the potential is updated simultaneously with the policy, then the shaping reward itself becomes non-stationary, and the standard invariance guarantee no longer applies directly. For this reason, if LenVM were to be used for RL shaping in future work, the most stable protocol would be to pretrain it first and then freeze it during policy optimization.

###### C.4. Why This is Preferable to a Sequence-Level Length Penalty

A common alternative is to add a sequence-level penalty such as Rtask − αL at the end of generation. The token-level formulation above has several advantages.

First, it yields dense credit assignment. Each additional token induces a local change in the length value, so the policy can receive an immediate signal about whether continuing the rollout appears worthwhile. In contrast, a terminal penalty must be propagated backward through the full sequence.

Second, it admits a matched value baseline. Since LenVM is trained directly on the return induced by rtlen, the resulting baseline is aligned with the optimization target for the length component. This reduces variance relative to using a terminal length penalty with a generic value function that must jointly absorb task quality and sequence length.

Third, it cleanly separates quality from efficiency. The task value network models the utility of the generated content, while LenVM models the expected future cost of additional decoding steps. This decoupling is especially natural when task reward is sparse or delayed.

Fourth, in the shaping setting, LenVM provides a structured progress potential rather than only a final scalar penalty. This is useful precisely because the reward is changed in a way that preserves the original optimum while making intermediate decisions easier to evaluate during training.

Finally, the trade-off coefficient remains interpretable. In the additive objective formulation, s acts directly on a length-specific advantage term. In the shaping formulation, β controls only the strength of the auxiliary progress signal. These two roles should be distinguished, since one changes the optimized objective while the other changes only the learning dynamics.

###### C.5. Connection to the Main Paper

The role of this section is to clarify that the LenVM objective is compatible with PPO-style RL because it already defines a proper discounted value-learning problem over decoding states. More importantly, the same learned signal admits two conceptually distinct uses in RL. It can be treated as a length-specific value function and directly incorporated into the policy objective when one wishes to optimize quality-efficiency trade-offs. Alternatively, it can be converted into a potential function for reward shaping under the standard fixed-potential assumptions when one wishes to preserve the original task objective while providing denser token-level learning signals.

This RL perspective also helps explain why LenVM is useful at inference time. Even outside RL, the model provides a prefix-conditioned estimate of how close generation is to termination. That signal can be exploited for length-controlled decoding and performance-efficiency trade-offs at test time. We do not claim empirical RL gains in this paper, and we leave RL fine-tuning with these formulations to future work.

0.08

| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |= =|0.997 0.99|98|
| | | | | | | | | | | | | | | | | | | | | | | | |1/|k|=|0.0078|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | |

f(z, k) with k = 128

l = 1, = 0.997

0.06

0.07

l = 1, = 0.9998

Relativeresolutionproxy||/ll

Relativeresolutionproxy||/ll

0.06

0.05

0.05

0.04

0.04

0.03

0.03

0.02

0.02

0.01

0.01

0.00

0.00

10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5 10.0 Logit z

100 101 102 103 104 105 Horizon length l

- Figure 7. Relative length resolution under finite precision from two complementary views. Left: the proxy f(z, k) as a

function of the logit z. Right: the same quantity expressed as a function of the horizon l using σ(−z) = γl. Markers indicate the finite-logit locations corresponding to l = 1 for representative values of γ.

#### D. Finite-Precision Analysis of Relative Length Resolution

LenVM is trained to regress remaining horizons over a wide dynamic range, from 1 to 32k. In this setting, the central numerical question is what relative length resolution the representation can maintain under finite precision. Since longer horizons naturally tolerate larger absolute deviations, the more relevant quantity is the relative perturbation ∣δl∣/l.

Recall that LenVM predicts vˆ = −σ(z) and that the length estimate is recovered from

1 + vˆ = γlˆ. (33) Since 1 + vˆ = 1 − σ(z) = σ(−z), we have

ln(σln(−γz))

, (34) or equivalently,

lˆ =

σ(−z) = γl. (35) Thus, the logit corresponding to horizon l is

1 − γl γl

z(l) = ln

. (36)

A first-order perturbation analysis with respect to z gives

≈ m(z)∣δz∣, (37)

δl l

where

σ(z) ln(σ(−z))

m(z) = −

. (38)

To obtain a lightweight finite-precision proxy, we model the local logit perturbation as

∣δz∣ ≈ ∣kz∣

, (39)

where k characterizes the effective precision level. In particular, taking k ≈ 2p for a floating-point format with p effective mantissa bits gives k = 128 for bfloat16, k = 1024 for fp16, and k = 8388608 for fp32. Substituting this approximation into the expression above yields

σ z)∣z∣ k ln(σ(−z))

≈ f(z, k) = −

δl l

. (40) We interpret f(z, k) as a proxy for the local relative length resolution under finite precision. For long horizons, corresponding to z → +∞, we have

σ(z) → 1, ln(σ(−z)) ≈ −z, (41) and therefore

1 k

δl l

. (42)

≈

Thus, at large horizons the representation approaches an approximately constant relative resolution floor. In contrast, the variation in relative resolution is concentrated in the short-horizon regime, where the corresponding logits remain finite and finite-precision effects are more pronounced. In practice, this means that the dominant numerical error arises in the small-length region rather than in the long-horizon tail.

Figure 7 summarizes this analysis from two complementary views. The left panel shows the relative resolution proxy f(z, k) directly in logit space, while the right panel rewrites the same quantity as a function of the horizon l using σ(−z) = γl. These plots show that the relative error is primarily concentrated in the small-l region, which is consistent with the empirical behavior observed in Figure 6. Together, they illustrate how finite-precision effects in logit space translate into horizon-dependent relative resolution.

#### E. Future-Dependent Weighting Changes the Regression Target

We clarify the population objective underlying LenVM training and show that weights depending on future rollout outcomes shift the regression target away from the state-conditional mean of the return proxy.

Setup. Let τ denote a sampled prompt-completion trajectory with generated length L(τ). For each nonterminal step t ∈ {0, . . . , L(τ) − 1}, define the per-token squared error

ℓθ(τ, t) ≜ (Vθ(sτt ) − Gtτ)2, Gtτ = −(1 − γL(τ)−t). (43)

- E.1. General Weighted Objective Consider the general weighted population objective

L(τ)−1

wt(τ) ℓθ(τ, t)]

Eτ,t[

∑

𝒥w(θ) =

, wt(τ) ≥ 0, (44)

t=0

L(τ)−1

wt(τ)]

Eτ,t[

∑

t=0

with empirical estimator

L(n)−1

wt(τ(n)) ℓθ(τ(n), t)

N

∑

∑

t=0

n=1

. (45)

ℒw =

L(n)−1

wt(τ(n))

N

∑

∑

t=0

n=1

Optimal predictor. Introducing the indicator I(sτt = s) and optimizing V(s) pointwise, the first-order condition gives

L(τ)−1

wt(τ)I(sτt = s) ⋅ (V(s) − Gtτ)] = 0. (46)

Eτ,t[

∑

t=0

Solving for V(s) yields the minimizer of 𝒥w:

Eτ,t[wt(τ) Gtτ ∣ sτt = s] Eτ,t[wt(τ) ∣ sτt = s]

Vw∗(s) =

, (47)

i.e., a w-weighted conditional mean of Gtτ at state s. Whether this coincides with the unweighted conditional mean Eτ,t[Gtτ ∣ sτt = s] depends on the structure of wt(τ).

###### E.2. Instantiations

- Case 1: Token-uniform weighting (wt(τ) = 1). Setting wt(τ) = 1 recovers the token-averaged objective

𝒥tok(θ) =

Eτ[

L(τ)−1

∑

t=0

ℓθ(τ, t)]

Eτ[L(τ)]

, (48)

where every non-terminal decoding step receives equal weight. Since wt(τ) = 1 is constant, it cancels in Eq. (47), and the optimal predictor is

Vtok∗ (s) = Eτ,t[Gtτ ∣ sτt = s]. (49) This is the state-conditional mean of the return proxy, which is exactly the quantity LenVM aims to estimate.

- Case 2: State-dependent weighting (wt(τ) = c(sτt ), c(s) > 0). If wt(τ) depends only on the current state sτt , then conditioning on sτt = s makes wt(τ) a deterministic constant c(s). By the linearity of conditional expectation, c(s) factors out of both the numerator and denominator in Eq. (47) and cancels:

c(s)Eτ,t[Gtτ ∣ sτt = s] c(s)

Vw∗(s) =

= Eτ,t[Gtτ ∣ sτt = s]. (50)

Such weights only rebalance the relative importance of different states in the global objective; they do not alter the per-state regression target.

- Case 3: Future-dependent weighting. If wt(τ) depends on outcomes beyond sτt , then conditioned on sτt = s, the weight wt(τ) remains a non-trivial random variable that may be correlated with Gtτ. In general,

Vw∗(s) ≠ Eτ,t[Gtτ ∣ sτt = s], (51)

meaning the regression target itself is shifted. A concrete example is trajectory-level averaging, which sets wt(τ) = 1/L(τ):

L(τ)−1

ℓθ(τ, t)]. (52)

𝒥traj(θ) = Eτ[

1 L(τ)

∑

t=0

Because L(τ) is determined only after the full rollout, 1/L(τ) is correlated with Gtτ conditioned on sτt = s: trajectories with larger L(τ) receive smaller weight yet produce more negative Gtτ. Consequently, 𝒥traj does not optimize the token-uniform objective in Eq. (48); it optimizes a future-length-reweighted variant with a shifted per-state target.

For LenVM, whose goal is dense token-level Monte Carlo regression over observed decoding states, we therefore adopt the token-averaged objective in Eq. (48).

#### F. Why Inverting the Transformed Horizon Underestimates Expected Remaining Length

In Section 4.4, we evaluate length prediction in the transformed space rather than by directly inverting the predicted horizon into a raw length estimate. This appendix explains why such inversion systematically underestimates the true expected remaining length.

Let L denote the random remaining generation length from a given decoding state, with L ≥ 0. Define the transformed horizon

u(L) = 1 − γL, (53)

where γ ∈ (0,1). Suppose the model predicts the conditional expectation in this transformed space exactly:

uˆ = E[u(L)]. (54) A natural raw length estimate is then obtained by inversion:

ln(ln1 −γuˆ)

Lˆ = u−1(uˆ) =

. (55)

We now show that this estimate is always less than or equal to the true expected remaining length E[L]. The first and second derivatives of u(L) are

u′(L) = −γL ln γ, (56) and

u′′(L) = −γL(ln γ)2. (57)

Because γ ∈ (0,1), we have ln γ < 0, so u′(L) > 0. Thus u(L) is strictly increasing. Also, since γL > 0 and (ln γ)2 > 0, we have u′′(L) < 0 for all L. Therefore, u(L) is strictly concave.

By Jensen’s inequality for concave functions,

E[u(L)] ≤ u(E[L]). (58)

Since u(⋅) is strictly increasing, its inverse u−1(⋅) is also strictly increasing. Applying u−1 to both sides of Equation 58 preserves the inequality:

u−1(E[u(L)]) ≤ u−1(u(E[L])) = E[L]. (59) Therefore,

Lˆ ≤ E[L]. (60)

As a result, the strict convexity of the discounted return mapping, combined with Jensen’s inequality, guarantees that the inverted length Lˆ is bounded above by the true expected length E[L]. The equality Lˆ = E[L] holds if and only if the variance of L is zero (i.e., the remaining length is completely deterministic). In all stochastic sequence generation scenarios where the remaining length has variance, the inversion will systematically underestimate the true expected length. In other words, even if the model predicts the conditional mean accurately in the transformed value space, directly mapping that prediction back to the raw length space via u−1 yields an underestimate of the true expected remaining length. The underestimation issue is strictly a consequence of converting to the length space; there is no inherent problem when operating directly within the value space.

#### G. Derivation of the Exponential Tilting Solution in the Performance–Efficiency Tradeoff Experiment

We derive the closed-form solution to the KL-regularised length-steering objective in Eq. (11). At each decoding step t, let p(x) be the base model’s distribution over a finite candidate set 𝒱t and let vˆ(x) be the LenVM’s value prediction for token x. The objective has two goals: (1) minimise the expected LenVM value to steer generation toward shorter completions, and (2) penalise large deviations from the base model via KL divergence to preserve generation quality. We seek the distribution p′ that solves, for β < 0:

Ep′[vˆ(x)] −

DKL(p′∥p) subject to ∑

p′(x) = 1, p′(x) ≥ 0. (61)

1 β

min

p′

x

Since β < 0, the term −β1 > 0, so the KL divergence enters with a positive coefficient, making the objective strictly convex in p′ and guaranteeing a unique global minimiser. (If β > 0 were used instead, the KL term would be subtracted, rendering the objective unbounded below and the problem ill-posed.)

Expanding the KL term and introducing a Lagrange multiplier λ for the normalisation constraint, the Lagrangian is

p′ x p(x)

p′(x)ln

ℒ(p′, λ) = ∑

p′(x) vˆ(x) −

p′(x) − 1). (62)

− λ(∑

1 β

∑

x

x

x

Taking the derivative with respect to p′(x) and setting it to zero:

p′ x p(x)

vˆ(x) −

1 β(ln

+ 1) − λ = 0. (63)

Solving for p′(x):

p′(x) = p(x) exp(β vˆ(x) − 1 − βλ). (64)

The normalisation condition ∑x p′(x) = 1 fixes the constant, giving the final result:

p(x) exp(β ⋅ vˆ(x)) ∑x′∈𝒱t p(x′) exp(β ⋅ vˆ(x′))

p′(x) =

, β < 0. (65)

This is the Gibbs / softmax form, and it is the unique global minimiser. With β < 0, tokens with lower predicted values (shorter expected horizons) receive higher probability mass, steering generation toward shorter completions while ∣β∣ controls how far the distribution drifts from the base model.

