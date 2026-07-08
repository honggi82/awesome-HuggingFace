# arXiv:2603.08462v2[cs.LG]18May2026

## Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck

Fabio Valerio Massoli∗ Qualcomm AI Research†

Andrey Kuzmin Qualcomm AI Research†

Arash Behboodi Qualcomm AI Research†

#### Abstract

Chain-of-Thought (CoT) prompting improves LLM accuracy on complex tasks but often increases token usage and inference cost. Existing “Budget Forcing” methods reduce cost via fine-tuning with heuristic length penalties, suppressing both essential reasoning and redundant filler. We recast efficient reasoning as a lossy compression problem under the Information Bottleneck (IB) principle, and identify a key theoretical gap when applying naive IB to transformers: attention violates the Markov property between prompt, reasoning trace, and response. To resolve this issue, we model CoT generation under the Conditional Information Bottleneck (CIB) principle, where the reasoning trace Z acts as a computational bridge that contains only the information about the response Y that is not directly accessible from the prompt X. This yields a general Reinforcement Learning objective: maximize task reward while compressing completions under a prior over reasoning traces, subsuming common heuristics (e.g., length penalties) as special cases (e.g., uniform priors). In contrast to naive token-counting approaches, we introduce a semantic prior that measures token cost by surprisal under a language model. Crucially, the prior is queried only for token-level log-probabilities, adding negligible overhead to the training loop. Empirically, our CIB objective prunes reasoning redundancy while preserving fluency and logic, improving accuracy at moderate compression and enabling aggressive compression with minimal accuracy drop. These gains generalize across model families and task domains, confirming CIB as a domain-agnostic CoT compression framework.

#### 1 Introduction

Chain-of-Thought (CoT) prompting [45] is the primary mechanism for unlocking reasoning in Large Language Models (LLMs), allowing models to allocate test-time computation for complex tasks. However, this gain incurs significant costs: reasoning chains are often excessively verbose, increasing latency and compute usage. Consequently, “Budget Forcing”—constraining models to yield correct answers within a restricted token budget—has emerged as a critical frontier in efficient inference. Current approaches relying on naive length penalties or strict training-time length constraints are suboptimal. Whether penalizing output length or enforcing a hard token limit, these methods impose a uniform cost on every token, implicitly assuming all tokens contribute equally to the solution. This uniform token penalty—effectively a “flat tax” on computation—ignores the distinction between essential reasoning steps and redundant fillers. Optimizing under such a metric is brittle: models are incentivized to delete tokens regardless of semantic relevance, discarding crucial intermediate logic to satisfy the budget. This makes the accuracy–compute trade-off difficult to tune, as a single weight (or limit) may over-penalize hard prompts while under-penalizing redundancy in easy ones. In this work, we reframe efficient reasoning not as token minimization, but as lossy compression. We propose a unified framework based on the Information Bottleneck (IB) principle [38], positing

∗Corresponding author: fmassoli@qti.qualcomm.com †Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.

Preprint.

0.0

0.38

| | | | | | | |
|---|---|---|---|---|---|---|
| |CI|B Q =1.<br><br>|=5 5B<br><br>|.e-5| | |
| | | | |L1-Ex|act<br><br>| |
|CIB|Q =7B<br><br>|=1<br><br>|.e-4| |DLER|-1.5B|
| |=1.e-3<br><br>|=5.e-4<br><br>| | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
|m|z logQ (z ean<br><br>|)| | |
|| |
|---|
<br><br>±|1 std| | | |

AIME24Acc.(Pass@1)

0.2

0.36

MinimalityReward

0.4

0.34

0.6

0.32

0.30

0.8

0.25

| |L3L1-E|XACT| | | | |
|---|---|---|---|---|---|---|

1.0

1.2

0.24

0 1000 2000 3000 4000 5000

2400 2600 2800 3000 3200 3400 3600 3800

Mean Completion Length

Mean Completion Length

- Figure 1: Left: Minimality Reward vs. Completion Length. We observe a consistent negative correlation between the completion length and the minimality reward. The shaded blue region shows the ±1σ band representing the spread of the information cost for the token chosen within CoTs with similar length. Right: Accuracy vs. CoT Pareto Frontier. The β parameter confers fine-grained control over the accuracy-compression trade-off. 7B prior (■) yields stronger compression than a

###### ■

1.5B prior (•). Baselines: DLER [22] (⋆), L3L1-EXACT [1] (

), and our L1-Exact‡ model (▼).

that an ideal reasoning chain is the minimal sufficient statistic of the prompt required to predict the answer. We identify that standard IB [38] cannot be naively applied to transformers due to a theoretical inconsistency we term the “Attention–Markov Incompatibility”: the attention mechanism grants the decoder direct access to the prompt, violating the Markov chain assumption (Y ↔

- X ↔ Z) required by standard IB. We resolve the problem by modeling CoT generation under the Conditional Information Bottleneck (CIB) as Source Coding with Side Information. As a result, a novel Reinforcement Learning (RL) objective naturally arises from the CIB framework. Instead of a uniform length penalty, we assign a semantic cost to each token based on its information content relative to a frozen base model. This formulation aligns cost with information flow: the model is encouraged to “pay” for informative tokens that increase answer probability while suppressing redundancy. Empirically, this allows for precise navigation of the Pareto frontier, achieving a superior accuracy–compression trade-off compared to length-based baselines (Figure 1, right). Our contributions are as follows:

- • We identify the limitations of length-based budget forcing, showing that uniform penalties and hard limits conflate essential reasoning with redundancy.
- • We propose a theoretical framework resolving the “Attention–Markov Incompatibility” via the CIB, yielding a semantic token cost based on relevance rather than length.
- • We demonstrate that this formulation compresses reasoning traces while achieving Pareto optimal accuracy-compression trade-off.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 formalizes the “Attention–Markov Incompatibility", derives the CIB objective, and presents our reward model. Section 4 proves that standard length-based penalties arise as a special case of CIB under non-informative priors. Section 5 reports experimental results. Section 6 we report our conclusions and briefly discuss limitations.

#### 2 Related Work

Budget Forcing and Efficient Reasoning. Recent studies suggest that optimal reasoning compute should scale with problem complexity [51], yet unconstrained models often exhibit excessive verbosity even on simple tasks [31]. This has motivated “Budget Forcing” strategies spanning training and inference, including reward shaping with length costs [1], token-level advantages [10], and hard truncation [22]. More granular approaches include difficulty-aware allocation [5] and referenceguided budgeting [46, 20, 23], monotonic uncertainty reduction [44], sometimes tracking history [12] or decomposing costs per-token [15]. Inference-only methods steer generation via auxiliary predictors [19, 8] or employ early-exit decoding [26, 42]. Alternative paradigms replace verbose CoT with concise drafting [49, 33], selective reasoning policies [40], or trace compression via token

pruning and skipping [48, 6, 7, 4]. [41] further propose budget-aware evaluation metrics. Several more recent works consider budget forcing in the context of multi-turn reasoning tasks [14, 35, 25]. While effective, these methods largely rely on naive token counts as a cost proxy. In contrast, we ground budget forcing in information theory, penalizing tokens based on semantic surprisal rather than raw length.

Information Theory in Large Language Models. The IB principle [38] was proposed as a framework for analyzing deep learning [37], followed by various discussions [34], applications in reasoning and robustness [16], and hallucination detection [39]. However, these works differ from ours in two key respects. First, their objectives typically target generalization or explainability of deep learning rather than strict computational efficiency of reasoning models. Second, they apply the standard IB formulation, which assumes a Markov chain where the latent representation Z mediates all information. Instead, we explicitly take into account the structure of transformer architectures, where the attention mechanism grants the decoder direct access to the prompt, X, creating a collider structure (X,Z) → Y which breaks the aforementioned Markov property. To the best of our knowledge, this work is the first to unify “Budget Forcing” and Information Theory under a Conditional Information Bottleneck framework.

#### 3 Methodology

We now formalize efficient reasoning as an optimization problem within the CIB framework. We report here the most relevant equations and we refer the reader to Appendix A for the full derivation. In what follows, we refer to X, Z, and Y , as the prompt, CoT, and ground truth answer, respectively.

The Attention–Markov Incompatibility. The standard Information Bottleneck (IB) principle [38] seeks a representation Z that maximally compresses the input X while preserving information about the target Y . Formally, it minimizes the Lagrangian:

LIB = I(X;Z) − µI(Y ;Z) (1) over P(Z|X) where µ controls the trade-off between compression (minimizing mutual information

- I(X;Z)) and prediction (maximizing I(Y ;Z)). Crucially, the standard IB assumes the Markov chain Y ↔ X ↔ Z, implying that Z is the sole channel through which information flows from X to Y . However, this assumption is fundamentally violated in transformer-based Large Language Model (LLM)s. Due to the causal attention mechanism, the decoder predicting Y attends to both the prompt X and the generated chain Z. This forms a collider structure: (X,Z) → Y . We term this inconsistency the Attention–Markov Incompatibility. Under the standard IB objective, maximizing
- I(Y ;Z) can be inefficient as it ignores that the model has access to the query X during the answer generation. This can lead to keeping redundant information about the query X. It is important to note that the conditional probability P(Y |X) of the answer given the query is unknown, and exactly what we want to simulate using the intermediate reasoning trace Z.

Conditional Information Bottleneck for Reasoning. To resolve the Attention–Markov Incompatibility, we propose grounding “Budget Forcing” in the Conditional Information Bottleneck (CIB). We view the prompt X as side information that is always available to the answer generator. We require Z to encode only the additional information necessary to predict Y given X. The objective becomes:

LCIB = I(X;Z) − µI(Y ;Z|X) (2)

Minimizing I(X;Z) (or a related upper bound on the rate) while maximizing the conditional predictive power I(Y ;Z|X) ensures that the chain Z is penalized for redundancy with X but rewarded for explaining Y . We use an LLM πθ(·|·) to re-parameterize the optimization problem.

3.1 Problem Formulation We consider a reasoning task defined by a dataset distribution PD(X,Y ), where X is the prompt and

- Y the ground truth answer. We aim to learn a stochastic policy πθ(Z | X) that generates a CoT Z to bridge the gap between X and Y , while πθ(Y | X,Z) generates the correct answer. Our goal is to optimize the policy πθ to maximize the Sufficiency of Z for predicting Y , while minimizing the Minimality (information cost) of Z relative to the side information X. This is

CIB Qϕ =7B (ours)

| |
|---|

| |
|---|

DLER-1.5B (baseline)

MATH500

###### AIME24

###### AIME25

Minerva

0.0010

0.0004

0.0004

0.0004

p.d.f.

0.0005

0.0002

0.0002

0.0002

0.0000

0.0000

0.0000

0.0000

0 2500 5000

0 2500 5000

0 2500 5000

0 2500 5000

Mean Length

Mean Length

Mean Length

Mean Length

- Figure 2: Lengths Distribution. Compared to the baseline length distribution (blue curve), the minimality term shifts the length distribution towards shorter completions (green curve). The plotted distributions correspond to models with similar accuracy (see Table 1).

formalized by the CIB objective: minθ LCIB(θ) = minθ I(X;Z) − µI(Z;Y | X), where µ ≥ 0 controls the rate-distortion trade-off. To derive our final reward function, we rewrite the previous objective as a maximization problem, rather than a minimization one. Therefore, our objective becomes:

(3)

LCIB(θ) = max

###### I(Z;Y | X)

###### −β I(X;Z)

max

θ

θ

Sufficiency

Minimality

where β gives direct control on the trade-off between accuracy and compression level (Figure 1, right). We refer the reader to Appendix A for the detailed discussion on the derivation.

Deriving the Sufficiency Term (Accuracy Reward). We aim to maximize the conditional Mutual Information (MI) I(Z;Y | X). We can write it as a function of the policy πθ(y|x,z),πθ(z|x) as:

πθ(y|x,z) P(y|x)

###### I(Y ;Z|X) =

P(x,y)P(z|x,y)log

x,y,z

πθ(y|x,z) P(y|x)

πθ(y|x,z) P(y|x)

P(x,y)πθ(z|x)

log

=

x,y,z

πθ(y|x,z) P(y|x)

≥

P(x,y)πθ(z|x)log

,

x,y,z

where we used the inequality xlog x ≥ log x in the last step. Note that the mutual information

- I(Z;Y | X) can be decomposed as H(Y | X) − H(Y | X,Z). The first term H(Y | X) represents the inherent difficulty of the dataset and is constant with respect to θ. Thus, maximizing sufficiency is equivalent to minimizing the conditional entropy H(Y | X,Z). We can maximize the lower bound on it and approximate it further using the query-answer samples (xi,yi). The first

term of the optimization problem can then be approximated as: mi=1 EZ∼π

θ(Z|xi)[log πθ(yi|xi,Z)], where m is the number of samples. In many cases, like RLVR, a verifier Qρ(yi|xi,z) is used to score the answer. Therefore, we can also optimize the following variational lower bound:

m i=1 EZ∼π

θ(Z|xi)[log Qρ(yi|xi,Z)]. See Appendix A for the details of our derivation. In our experiments, we choose Qρ(Y |X,Z) such that it gives a reward of 1 (0) for correct (wrong) answers. From Log-Verifier to a Binary Accuracy Reward. Our variational surrogate for sufficiency uses a verifier score log Qρ(yi | xi,zi). In our setting the verifier is deterministic, returning Qρ(y | x,z) ∈ {0,1} (1 if the extracted answer is correct, else 0), so the log-score is ill-defined for incorrect answers. We therefore use the ε-smoothed verifier Qρ(y | x,z) := ε+(1−ε) ( y(x,z) = y) , where ε ∈ (0,1) and y is the predicted answer. Then log Qρ(y | x,z) = log ε − log ε ( y(x,z) = y) . Since log ε is a constant w.r.t. (θ,z) and −log ε > 0, maximizing E[log Qρ(y | x,z)] is equivalent (up to an affine transformation) to maximizing E[( y(x,z) = y)] . Accordingly, we define the accuracy reward as: racc(x,y,z) := ( y(x,z) = y), which is a finite, stable surrogate for the log-verifier objective.

Deriving the Minimality Term (Information Cost). We aim to minimize the MI I(X;Z) to penalize redundancy in the CoT: I(X;Z) = EX,Z log π

θ(Z|X) P(Z) .

However, computing P(Z) is not tractable. Therefore, we introduce an unconditional variational prior Qϕ(Z) (a distribution over Z that does not observe X) to find a variational bound similar to [3].

πθ(Z | X) P(Z)

πθ(Z | X)Qϕ(Z) P(Z)Qϕ(Z)

I(X;Z) = EX,Z log

= EX,Z log

(4)

πθ(Z | X) Qϕ(Z) − DKL(P(Z) ∥ Qϕ(Z))

= EX,Z log

≥0

Dropping the non-negative KL term gives the upper bound: I(X;Z) ≤ EX,Z [−log Qϕ(Z)]−H(Z | X), where Z ∼ πθ(·|X). We instantiate Qϕ(Z) using a frozen, pre-trained base model (not an instruction-finetuned model), ensuring it captures the statistics of general language without taskspecific conditioning. More details on the derivation can be found in Appendix A.

The first term, EX,Z[−log Qϕ(Z)], represents the cross-entropy rate (or description length) of the chain under the prior. It corresponds to the expected value of the reasoning trace information cost:

C(Z) := |tZ=1| −log Qϕ(zt | z<t).

The second term, −H(Z | X), corresponds to the negative entropy of the policy. In RL algorithms like PPO, this term is naturally handled via an entropy regularization bonus to encourage exploration.

Reward Modeling. Combining the bounds, we aim to maximize the following objective:

LCIB = E(X,Y )∼P

D,Z∼πθ log Qρ(Y |X,Z) + β

T

log Qϕ(zt | z<t) , (5)

t=1

where the first term represents the accuracy score from the verifier, Qρ, and Qϕ is chosen as prior distribution. We define our reward model as: R(X,Y,Z) := racc(X,Y,Z) + βrmin(X,Z), where

racc(X,Y,Z) := Y (X,Z) = Y is the accuracy reward, taking a value of 1 if the predicted answer matches the ground truth Y , and 0 otherwise, and rmin(X,Z) := Tt=1 log Qϕ(zt | z<t),is the cumulative surprisal (information cost) of the reasoning chain relative to the prior. Accuracy remains the primary objective, while rmin acts as a semantic regularizer controlled by the coefficient β. This effectively assigns a semantic cost to every token: the cost −log Qϕ penalizes low-probability (high-surprisal) tokens unless they contribute significantly to solving the task (racc). Tokens that are redundant or verbose increase the cumulative cost without improving accuracy, and are thus suppressed by the policy.

#### 4 Theoretical Analysis: A Unified Framework

A central motivation for this work is demonstrating that the CIB serves as a general framework from which, e.g., length-based penalties naturally arise as a special case. As an example, we prove that length-constrained methods correspond to the CIB rate term with non-informative priors.

- Proposition 4.1. A standard length-based penalty (e.g., g(Z) = αf(|Z|)) is equivalent to the CIB objective under the assumption of a maximum entropy (uniform) prior, Q, over the vocabulary.

Proof. Let |V | be the vocabulary size and consider the minimality term −log Q(zt). A Maximum Entropy prior implies a uniform distribution over the vocabulary V (i.e., Q(zt) = |V1| for all zt). Thus, the surprisal of every token becomes constant: c = log |V |. Then, the total information cost for a CoT, Z, of length T becomes: −logQ(Z) = − Tt=1 log(|V1|) = T · log|V |. Substituting this into the CIB objective, the penalty term becomes βTlog|V |. By setting α = βlog|V |, we recover a linear length penalty. This proves that linear penalties implicitly assume that all tokens carry equal information content (log|V |), ignoring the underlying semantics of the CoT.

| |
|---|

- Proposition 4.2. Target-length penalties, such as LCPO-Exact [1], correspond to the CIB objective with a Laplace prior.

Proof. Any penalty function g(Z) applied to the reward can be interpreted as an implicit prior Q(Z) ∝ exp(−g(Z)). LCPO-Exact penalizes deviation from a target length ngold via the term

g(Z) = |ngold − ny|, where ny is the length of the generated CoT. The corresponding implicit prior is: QLCPO(Z) ∝ e−|n

gold−ny|.

This is a Laplace-like distribution over the sequence length T, centered at ngold. Interpreting LCPO through this lens reveals a strong inductive bias: it posits that there exists a golden length for reasoning length, and any deviation (shorter or longer) is exponentially improbable.

| |
|---|

Crucially, in both Propositions 4.1 and 4.2, the implicit prior Q(Z) depends solely on the sequence length, whereas our proposed CIB method uses a language model prior defining a per-token cost.

#### 5 Experimental Results

Training and Evaluation Details We evaluate CIB on three small reasoning model families, namely, DLER-{1.5B, 7B} [22], OpenReasoning-Nemotron-1.5B [2, 30], and DeepScaleR-1.5B [24]. We focus on small-scale models (1.5B–7B parameters) as they represent the primary deployment target for efficient reasoning under latency and memory constraints. Training uses GRPO [36] with a group size of 16 and a Qwen2.5-Base-{1.5B, 7B} prior; the prior is used at training time only, adding no cost at inference. Our main evaluation spans five mathematical reasoning benchmarks: MATH500 [21], AIME24 [28], AIME25 [29], Minerva [17], and OlympiadBench [9]. Math provides unambiguous, deterministic targets, serving as a controlled evaluation setting. To demonstrate that CIB generalizes beyond math, we report in Appendix B evaluations results on GPQA [32], MMLUPro [43], and LiveCodeBench [13]. We follow the evaluation protocol of [22], using vLLM as the inference engine (temperature 0.6, topp = 0.95, max tokens 32K, 16 generations per prompt) and report pass@1 accuracy. Further training and evaluation details are in Appendix D. We provide the list of assets used in the paper in Appendix F.

###### 5.1 Main Results

We begin by verifying that the minimality reward (Section 3) provides a usable learning signal. As shown in Figure 1 (left), it exhibits a negative correlation with completion length, yet its limited dispersion indicates sensitivity to the token sequence rather than to length alone. Table 1 summarizes accuracy and average CoT length across five mathematical reasoning benchmarks. For each model family, we evaluate CIB at two compression strengths (β−=5×10−5, β+=1.5×10−4), selected on the training distribution to target mild and “more aggressive” compression, and two prior sizes (1.5B and 7B). For DeepScaleR-1.5B we include multiple external baselines, since available, to provide a broader comparison: L3L1-EXACT [1], evaluated using the authors’ publicly available checkpoint, and ThinkPrune-I2k [11], HAPO [12], and LAPO-I [46] (marked with §), whose results are taken directly from the literature [46]—hence some benchmark entries are not available. Concerning our models, full results with standard error of the mean (SEM) are in Tables 3–5 in Appendix B. Moreover, a comparison with additional methods is reported in Appendix E.

CIB improves accuracy while compressing reasoning. Across all four model groups, CIBβ− improves average accuracy over the uncompressed baselines while simultaneously reducing CoT length (Table 1). DeepScaleR-1.5B provides the broadest comparison against length-based penalties from the literature. Here, CIBβ− increases avg. accuracy by 1.5% while reducing avg. CoT length by ∼ 24%, with gains consistent across both easy and hard benchmarks. In contrast, length-based penalty variants often incur accuracy degradation—most visibly, L3L1-EXACT drops AIME24 accuracy by 13.8% despite achieving strong compression. Compared to ThinkPrune-I2k [11], CIB7B achieves higher accuracy and stronger compression on all shared benchmarks, while CIBβ−,1.5B further increases accuracy at the cost of slightly longer CoT. LAPO-I [46] generally sits between CIBβ−,1.5B and CIB7B on the accuracy–compression plane: CIBβ−,1.5B achieves higher accuracy on MATH500 and AIME24, while CIB7B compresses more aggressively. Although LAPO-I appears to improve upon CIB7B in accuracy, the error analysis in Appendix B shows that the two agree within uncertainty. Increasing β from β− to β+ smoothly trades a small amount of accuracy for additional compression—up to ∼1.6 points for 5%–21% further token reduction (Table 1, Avg. column). Compared to the 1.5B prior, the 7B prior unlocks deeper compression, yielding the shortest CoT in every group; the only exception is L3L1-EXACT on DeepScaleR-1.5B, whose stronger compression comes at the cost of substantial accuracy degradation (Table 1, red entries). These benefits generalize across all model families considered. On Nemotron-1.5B, CIBβ−,1.5B achieves

- Table 1: Accuracy (↑) and average CoT length (↓) across different benchmarks. Bold indicates best performance within each model group. Green highlights improved average accuracy or token count vs. the baseline; red marks significant degradation. CIB subscripts denote the compression setting (β− = 5 × 10−5, β+ = 1.5 × 10−4) and the prior size (1.5B or 7B). Error estimates are reported in Tables 3–5 in Appendix B. ∗: Pass@1 with n = 16 for consistency, the official HuggingFace page reports pass@1(Avg@64); †: Our models; ‡: Models we trained using the L1-EXACT length penalty [1]; §: Results from [46].

Model MATH500 AIME24 AIME25 Minerva Olympiad Avg.

Acc./Tok. Acc./Tok. Acc./Tok. Acc./Tok. Acc./Tok. Acc./Tok. DeepScaleR-1.5B

- Baseline 85.8/3190 38.2/9331 25.4/8907 19.9/6386 54.8/5922 44.8/6747 L3L1-EXACT 85.2/1932 24.4/2637 20.2/2608 19.8/2124 48.8/2369 39.7/2334 ThinkPrune-I2k§ 85.5/1707 34.9/5095 −/− −/− 54.7/3498 −/− HAPO§ 84.4/2370 31.4/7702 −/− −/− 51.4/4571 −/− LAPO-I§ 86.3/2168 38.1/5371 −/− −/− 56.3/4024 −/−

CIB†β

−,1.5B 87.6/2359 40.2/7298 27.8/6866 20.4/4510 55.5/4472 46.3/5101 CIB†β

+,1.5B 87.2/2017 38.2/6937 27.6/6757 20.1/4147 55.3/4211 45.8/4814 CIB†7B 86.1/1337 37.5/5003 25.0/5389 19.7/2058 55.0/3065 44.7/3370 Nemotron-1.5B Baseline∗ 85.8/8509 43.1/22931 35.1/24086 11.0/13011 43.8/16270 43.8/16961

- L1-Exact‡ 85.8/8807 40.2/22079 30.8/23243 11.4/12925 44.0/15316 42.4/16474

CIB†β

−,1.5B 86.8/7536 45.3/21962 34.1/22980 11.7/11118 52.7/15089 46.1/15737 CIB†β

+,1.5B 85.4/5982 41.5/20954 34.0/21407 10.9/10051 50.8/13894 44.5/14458 CIB†7B 86.7/5789 42.7/20547 32.3/20678 10.6/8752 51.1/12242 44.7/13602 DLER-1.5B Baseline 86.4/1873 33.4/3478 23.9/3290 20.9/2587 54.1/2766 43.7/2799

- L1-Exact‡ 86.6/1751 35.0/3274 22.7/3037 21.2/2277 54.0/2577 43.9/2583

CIB†β

−,1.5B 88.4/1687 35.4/3198 25.3/3260 21.6/2346 54.3/2616 45.0/2621 CIB†β

+,1.5B 87.6/1421 34.1/3124 24.2/3027 21.0/2305 54.1/2544 44.2/2484 CIB†7B 85.0/805 31.6/2504 22.4/2257 20.4/1327 51.6/1631 42.2/1705 DLER-7B

Baseline 93.6/1159 49.6/3221 36.6/3271 27.3/1849 65.0/2840 54.4/2468 L1-Exact‡ 92.8/675 46.2/2982 31.2/2564 26.1/1144 61.9/1872 51.6/1847

CIB†β

−,7B 94.0/1033 49.7/2642 37.1/2962 27.6/1381 64.1/2361 54.5/2076 CIB†β

###### +,7B 92.2/678 48.2/2419 35.4/2550 25.8/851 62.7/1704 52.9/1640

the largest accuracy gain (+2.3 points) with a ∼7% token reduction, while CIB7B reduces tokens by ∼20% while still improving over the baseline. CIB also scales to larger models: on DLER-7B,

CIBβ−,7B matches the baseline’s strong accuracy while reducing average CoT length by ∼16%, and CIBβ+,7B achieves roughly a one-third reduction with a 1.5-point accuracy trade-off.

Generalization beyond math. Although CIB is trained exclusively on math data, its benefits transfer to science, knowledge, and code benchmarks (Table 5 in Appendix B). On DeepScaleR-1.5B, CIBβ− improves accuracy on GPQA Diamond (+1.9%), MMLU-Pro (+2.2%) while matching LiveCodeBench, with token reductions between up to 38.6%. On DLER-7B, it achieves comparable performance with the baseline while compressing CoT up to 14.6%. These results indicate that the compression learned by CIB is not specific to mathematical reasoning but generalizes to broader reasoning and knowledge tasks.

1.15

1.2

### Golden Zone

DLER-1.5B (baseline)

1.0

1.10

CIB Qϕ =7B (ours)

Surprisal(nats)

AccuracyGain

CIB Qϕ =1.5B (ours)

0.8

1.05

Information Gain (Qϕ =7B)

0.6

Information Gain (Qϕ =1.5B)

1.00

0.4

0.95

0.2

0.90

0.0

0.0 0.2 0.4 0.6 Compression Factor

0 20 40 60 Token Position

- Figure 3: Left: Efficiency Gain. Efficiency gain of CIB across diverse benchmarks and models. Points falling in the upper half-plane (“Golden Zone") exhibit strictly superior efficiency, achieving

higher information density with reduced computational cost. Colors denote models: CIBDLERQ

ϕ=1.5B (•), CIBDLERQ

ϕ=7B (•), CIBDeepSc.Q

ϕ=1.5B (•), CIBDeepSc.Q

ϕ=7B (•). Shapes denote tasks: MATH500 (•), AIME24 (■), AIME25 (▲), Minerva (♦), Olympiad (▼). Right: Information Density Profile. Token-wise surprisal evaluated against the baseline language prior. Lower surprisal indicates predictable, lowinformation filler.

Effect of β on the Accuracy–Efficiency Trade-off. The β weight in the CIB objective (Equation 3) provides fine-grained control over the accuracy–compression trade-off. As illustrated in Figure 1 (right), sweeping β across its range traces a smooth Pareto frontier: lower values prioritize accuracy, while higher values aggressively compress the chain of thought. On AIME24, for instance, intermediate β settings retain the majority of the baseline accuracy while substantially reducing token count, whereas extreme values reveal a clear knee beyond which further compression degrades reasoning quality. This continuous knob allows practitioners to select the operating point that best suits their downstream latency or cost constraints.

Efficiency Gain. To quantify the trade-off between reasoning performance and computational cost, we define two metrics: the Compression Factor = 1 − ℓCIB/ℓbase, measuring the relative reduction in completion length, and the Accuracy Gain = ACIB/Abase, normalizing performance against the baseline. Figure 3 (left) visualizes these metrics across architectures and benchmarks. The upper half-plane (“Golden Zone”) represents models that are strictly superior in both speed and accuracy, while models in the bottom-right offer significant speedups where partial accuracy degradation is permissible. By filtering for the Golden Zone, we select models that are “smarter” and faster rather than simply truncated. Note that gains for each model are normalized to its own baseline. For visual clarity, Figure 3 (left) focuses on DLER-1.5B and DeepScaleR across the five math benchmarks and two different prior sizes.

Information Density. To understand how CIB compresses, we analyze the token-wise surprisal, −log p(zt | z<t,x), measured relative to a frozen reference model. As shown in Figure 3 (right), the baseline profile (dashed gray) exhibits valleys of low surprisal (≈ 0.1 nats), indicating highly predictable sequences carrying negligible unique information. In contrast, CIB profiles (solid blue and green) maintain a higher floor (≳ 0.2 nats): by penalizing cumulative low-utility transitions, the objective acts as a high-pass semantic filter, excising predictable filler while preserving high-entropy peaks. This confirms that CIB achieves compression not by random truncation, but by maximizing the average information rate, distilling reasoning traces to their essential computational content.

Qualitative CoT Comparison. To validate that our objective targets reasoning redundancy rather than essential logic, we analyzed reasoning traces across arithmetic and symbolic tasks. We observe that CIB systematically eliminates conversational scaffolding, redundant verification loops, and tautological checks. Unlike naive truncation, the semantic prior fundamentally alters the reasoning topology, preserving the “computational bridge” while filtering transitions that offer low marginal information regarding Y . Detailed case studies are provided in Appendix C (Figure 8–10).

Prior Overhead and Capacity Impact. It is important to understand whether the prior model represents a training bottleneck. Because Qϕ is frozen, it requires no gradient accumulation, backward passes, or optimizer states; it only adds a static VRAM footprint and a single forward pass over the generated trajectory, scoring all tokens in parallel. Using our distributed setup (4×H100; 8 generations per sample, vLLM backend), Table 2 compares wall-clock time for four prior sizes against a length-based penalty reward for DLER-1.5B. Scaling the prior up to 7B reduces rollout token throughput by less than 6% and the effective data collection rate by less than 6.2% (1.47→1.38 samples/sec), confirming that training is largely insensitive to prior size in wall-clock terms.

- Table 2: Training overhead of the semantic prior. Wall-clock comparison for DLER-1.5B. Average value and SEM over 100 steps.

Prior Step Time (s) Tokens/sec Samples/sec Length-based penalty (no prior model) 21.1 ± 0.3 3691 ± 92 1.47 ± 0.02

- Qϕ = 0.5B 21.8 ± 0.3 3677 ± 97 1.46 ± 0.03
- Qϕ = 1.5B 22.5 ± 0.3 3482 ± 96 1.42 ± 0.03 Qϕ = 3B 22.4 ± 0.4 3489 ± 99 1.43 ± 0.03 Qϕ = 7B 23.2 ± 0.4 3449 ± 98 1.38 ± 0.03

While the computational cost is benign, prior capacity matters for the quality of the semantic signal. If the prior is too weak, its predictive distribution flattens toward uniform and the semantic cost degenerates into a naive length penalty, unable to distinguish redundant tokens from critical reasoning steps. Similarly, a cross-family prior (e.g., Llama for a Qwen policy) introduces tokenizationdependent confounds: vocabulary and segmentation mismatches can penalize the policy’s native formatting rather than its true verbosity. Studying whether coarser-grained costs could mitigate this effect is an interesting direction for future work.

#### 6 Conclusions

In this work, we address efficient reasoning in LLMs by reframing “Budget Forcing” from an information-theoretic perspective. We identified the “Attention–Markov Incompatibility”—the structural inconsistency in applying standard Information Bottleneck principles to transformer architectures—and proposed a Conditional Information Bottleneck framework to resolve it. This yields a principled RL objective that subsumes existing length-based penalties as special cases (Propositions 4.1 and 4.2) while introducing a semantic token cost based on surprisal under a language model prior. Our experiments on reasoning benchmarks show that penalizing tokens by their semantic information content yields a more favorable accuracy–length trade-off than naive length penalties. By tuning β, one can traverse the Pareto frontier, achieving up to ∼ 48% token reduction with ≲ 1.5% accuracy degradation. We further find that stronger priors provide better redundancy estimates, enabling more aggressive compression with minimal performance loss—while introducing negligible training overhead (∼ 6% throughput reduction). These findings suggest that efficient inference requires moving beyond a “flat tax” on token count toward metrics that value computation based on its utility. More broadly, our framework’s central contribution is a general recipe for optimizing reasoning efficiency: by varying the implementations of the verifier and the prior, practitioners can explore a broad design space tailored to specific downstream tasks and deployment constraints.

Limitations. While our results demonstrate the effectiveness of CIB for compressing reasoning traces, some limitations should be acknowledged. The quality of compression depends on the reference prior Qϕ. As discussed in Section 5, a weak prior flattens toward uniform, degenerating the semantic cost into a naive length penalty. Furthermore, cross-family priors (e.g., a Llama prior for a Qwen policy) introduce tokenization-dependent confounds that we have not addressed. Studying coarser-grained cost functions that are robust to vocabulary mismatch is a promising direction. The regularization coefficient β and the prior capacity Qϕ jointly determine the operating point on the Pareto frontier. In our experiments, we used β values optimized for the 1.5B prior and did not re-tune when scaling to 7B, which may account for the slight accuracy gap observed. A systematic study of the (β,Qϕ) interaction could yield further improvements.

#### References

- [1] Pranjal Aggarwal and Sean Welleck. L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning. arXiv preprint arXiv:2503.04697, 2025.
- [2] Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding. arXiv preprint arXiv:2504.01943, 2025.
- [3] Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep Variational Information Bottleneck. In International Conference on Learning Representations, 2017.
- [4] Jeffrey Cheng and Benjamin Van Durme. Compressed Chain of Thought: Efficient Reasoning Through Dense Representations. arXiv preprint arXiv:2412.13171, 2024.
- [5] Zhengxiang Cheng, Dongping Chen, Mingyang Fu, and Tianyi Zhou. Optimizing Length Compression in Large Reasoning Models. arXiv preprint arXiv:2506.14755, 2025.
- [6] Sunguk Choi, Yonghoon Kwon, and Heondeuk Lee. CAC-CoT: Connector-Aware Compact Chain-of-Thought for Efficient Reasoning Data Synthesis Across Dual-System Cognitive Tasks. In Findings of the Association for Computational Linguistics: EMNLP 2025, 2025.
- [7] Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, and Qi He. Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.
- [8] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-Budget-Aware LLM Reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.
- [9] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.
- [10] Yinhar He, Yaochen Zhu, Mingjia Shi, Wendy Zheng, Lin Su, Xiaoqing Wang, Qi Guo, and Jundong Li. IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning. arXiv preprint arXiv:2602.19049, 2026.
- [11] Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.
- [12] Chengyu Huang, Zhengxin Zhang, and Claire Cardie. HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization. arXiv preprint arXiv:2505.11225, 2025.
- [13] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv preprint arXiv:2403.07974, 2024.
- [14] Neharika Jali, Anupam Nayak, and Gauri Joshi. Not All Turns Are Equally Hard: Adaptive Thinking Budgets For Efficient Multi-Turn Reasoning. arXiv preprint arXiv:2604.05164, 2026.
- [15] Shuyang Jiang, Yusheng Liao, Ya Zhang, Yanfeng Wang, and Yu Wang. Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling. arXiv preprint arXiv:2509.25827, 2025.
- [16] Shiye Lei, Zhihao Cheng, Kai Jia, and Dacheng Tao. Revisiting LLM Reasoning via Information Bottleneck. arXiv preprint arXiv:2507.18391, 2025.

- [17] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving Quantitative Reasoning Problems with Language Models. In Advances in Neural Information Processing Systems, volume 35, pages 3843–3857, 2022.
- [18] Cheuk Ting Li. Channel simulation: Theory and applications to lossy compression and differential privacy. Foundations and Trends® in Communications and Information Theory, 21(6):847–1106, 2024.
- [19] Junyan Li, Wenshuo Zhao, Yang Zhang, and Chuang Gan. Steering LLM Thinking with Budget Guidance. arXiv preprint arXiv:2506.13752, 2025.
- [20] Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, Kai Jia, and Zhifang Sui. SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning. arXiv preprint arXiv:2505.11274, 2025.
- [21] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step. arXiv preprint arXiv:2305.20050, 2023.
- [22] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. DLER: Doing Length pEnalty Right — Incentivizing More Intelligence per Token via Reinforcement Learning. arXiv preprint arXiv:2510.15110, 2025.
- [23] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning. arXiv preprint arXiv:2501.12570, 2025.
- [24] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. DeepScaleR: Surpassing O1-Preview with a 1.5b Model by Scaling RL. Notion Blog Post, 2025.
- [25] Siyuan Ma, Bo Gao, Xiaojun Jia, Simeng Qin, Tianlin Li, Ke Ma, Xiaoshuang Jia, Wenqi Ren, and Yang Liu. ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference. arXiv preprint arXiv:2602.23681, 2026.
- [26] Minjia Mao, Bowen Yin, Yu Zhu, and Xiao Fang. Early Stopping Chain-of-Thoughts in Large Language Models. arXiv preprint arXiv:2509.14004, 2025.
- [27] Mathematical Association of America. American Mathematics Competitions (AMC) 10/12

- 2023, 2023.

[28] Mathematical Association of America. American Invitational Mathematics Examination (AIME)

- 2024, 2024. Problems I and II.

[29] Mathematical Association of America. American Invitational Mathematics Examination (AIME)

- 2025, 2025. Problems I and II.

- [30] Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.
- [31] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple Test-Time Scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20286–20332, Suzhou, China, November 2025. Association for Computational Linguistics.
- [32] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv preprint arXiv:2311.12022, 2023.

- [33] Matthew Renze and Erhan Guven. The Benefits of a Concise Chain of Thought on ProblemSolving in Large Language Models. arXiv preprint arXiv:2401.05618, 2024.
- [34] Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, and David Daniel Cox. On the Information Bottleneck Theory of Deep Learning. In International Conference on Learning Representations, 2018.
- [35] Khushal Sethi. Don’t Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents. arXiv preprint arXiv:2604.08369, 2026.
- [36] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024.
- [37] Ravid Shwartz-Ziv and Naftali Tishby. Opening the Black Box of Deep Neural Networks via Information. arXiv:1703.00810 [cs], March 2017.
- [38] Naftali Tishby, Fernando C Pereira, and William Bialek. The Information Bottleneck Method. In Allerton Conference, 1999.
- [39] Jean-Francois Ton, Muhammad Faaiz Taufiq, and Yang Liu. Understanding Chain-of-Thought in LLMs through Information Theory. arXiv preprint arXiv:2411.11984, 2024.
- [40] Jiaqi Wang, Kevin Qinghong Lin, James Cheng, and Mike Zheng Shou. Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models. arXiv preprint arXiv:2505.16854, 2025.
- [41] Junlin Wang, Siddhartha Jain, Dejiao Zhang, Baishakhi Ray, Varun Kumar, and Ben Athiwaratkun. Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19916–19939, 2024.
- [42] Xi Wang, James McInerney, Lequn Wang, and Nathan Kallus. Entropy After </think> for Reasoning Model Early Exiting. arXiv preprint arXiv:2509.26522, 2025.
- [43] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.
- [44] Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, and Nancy F Chen. InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning. arXiv preprint arXiv:2603.17310, 2026.
- [45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
- [46] Xingyu Wu, Yuchen Yan, Shangke Lyu, Linjuan Wu, Yiwen Qiu, Yongliang Shen, Weiming Lu, Jian Shao, Jun Xiao, and Yueting Zhuang. LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization. arXiv preprint arXiv:2507.15758, 2025.
- [47] Aaron D Wyner and Jacob Ziv. The Rate-Distortion Function for Source Coding with Side Information at the Decoder. IEEE Transactions on Information Theory, 22(1):1–10, 1976.
- [48] Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. TokenSkip: Controllable Chain-of-Thought Compression in LLMs. arXiv preprint arXiv:2502.12067, 2025.
- [49] Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of Draft: Thinking Faster by Writing Less. arXiv preprint arXiv:2502.18600, 2025.
- [50] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2024.

###### [51] Junyu Zhang, Yifan Sun, Tianang Leng, Jingyan Shen, Ziyin Liu, Paul Pu Liang, and Huan Zhang. When Reasoning Meets Its Laws. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025.

#### A Conditional Information Bottleneck

Preliminaries. We denote the query, answer, and reasoning traces respectively by the random variables X,Y and Z. In this work, we assume that X,Y,Z ∈ X∗, where X∗ is the set of all finite sequences of tokens in the token space X. The underlying probability space of these random variables is given by (X∗,Σ∗), where Σ∗ is the co-product σ-algebra using the σ-algebras on each Xn. The space X is assumed to be discrete. The entropy of the random variable X, the conditional entropy of Y given X, and the mutual information between X and Y are defined as:

- H(X) := EX∼P

X

(−log P(X)),

- H(Y |X) := E(X,Y )∼P

(−log P(Y |X)), I(X;Y ) := H(X) − H(X|Y ).

XY

Formulation. Consider the query-answer pair (X,Y ). The function of reasoning is to generate Z such that the LLM probability πθ(Y |Z,X) is maximized. Budget forcing aims at compressing Z.

In the classical information bottleneck, this problem is formulated as maximizing the information gain I(Z;Y ) while minimizing the redundancy with I(X;Z), yielding the following optimization problem:

I(X;Z) − βI(Y ;Z),

min

πθ(Z|X)

under the Markov assumption Y ↔ X ↔ Z and the marginal probability constraint z πθ(z|x) = 1,∀x. In information bottleneck literature, the mutual information I(X;Z) is called rate or complex-

ity, while I(Y ;Z) is called relevance or information. We use the terms minimality and sufficiency in this paper.

In the context of reasoning, the Markov chain Y ↔ X ↔ Z does not hold, namely πθ(Y |X,Z) ̸= πθ(Y |X) because the dense attention mechanism breaks the Markov relation, and the response depends on both the query and the reasoning trace.

There is another subtle issue with the classical information bottleneck setup. The outcome of the optimization problem is πθ(z|x). The Markov property enables us to generate y based on z using p(y|z) = x p(y|x)p(x|z), which implicitly assumes the knowledge of the conditional probability p(y|x) at decoding time. Without Markov property, this relation cannot be used. Besides, p(y|x) is not available at the decoding time, and it is unknown in the context of reasoning. The goal of reasoning trace z is to enable the simulation of p(y|x) using πθ(z|x)πθ(y|z,x), which points toward a connection with channel simulation literature [18].

We would like to maximize the information gain of the reasoning trace Z with the knowledge of the query X. We can measure the gain using the conditional mutual information I(Y ;Z|X). The conditional information bottleneck version is as follows:

I(X;Z) − βI(Y ;Z|X),

min

P(Z|X,Y )

where we need the following marginal probability constraints to be satisfied:

P(z|x,y) = 1,∀x,y.

z

If we cast the problem as a maximization problem, the final optimization problem in terms of P(Z|X,Y ) is as follows:

P(z|x) P(z) s.t.

P(x,y)P(z|x,y)log P(y|x,z) − β

P(x)P(z|x)log

max

P(Z|X,Y )

x,y,z

x,z

P(z|x,y) = 1,∀x,y. (6)

z

The dependence on P(z|x,y) is implicit in various distributions like P(z|x),p(z) and p(y|x,z), and we can solve this problem using an iterative algorithm, similar to Blahut-Arimoto algorithm, proposed in [38]. Given that z is from the space of reasoning traces, it is not tractable to use the same approach.

We would like to remark that the information bottleneck problem is an instance of lossy compression under log-loss distortion metric. In this sense, the conditional information bottleneck is akin to lossy compression with side information, which was studied by Wyner and Ziv [47] under different settings.

Practical Implementation. The information bottleneck optimization in deep learning is directly intractable, and the approximate bounds are used for training such variational information bottleneck [3]. In practice, for training LLMs, we do not directly optimize over P(Z|Y,X) but rather optimize the model parameter θ.

The parameter θ controls the two probabilities πθ(z|x) and πθ(y|x,z), both the inference part of the AR generative model (instead of P(Z|Y,X). We solve the following optimization problem:

LCIB(θ) = I(Y ;Z|X) − βI(X;Z) s.t. P(y|x) =

max

θ

πθ(y|x,z)πθ(z|x), ∀x,y. (7)

z

The last constraint should be satisfied to have a valid probability distribution. Since all the probabilities πθ(·|·) are parametrized to sum up to one, we do not need to explicitly add the constraint.

Note that we can write P(z|x,y)P(y|x) = πθ(y|x,z)πθ(z|x). In other words, we can obtain a valid P(z|x,y) from πθ(y|x,z) and πθ(z|x), and vice versa. Therefore, the above optimization problem is just a reparameterization of the original information bottleneck problem and yields the same optimal value.

There are some challenges with the above optimization problem. First, the conditional distribution P(y|x) is unknown in general, and we have access to it only through the samples. Second, we should approximate the information theoretic quantities, namely the sufficiency term I(Y ;Z|X) and the minimality term I(X;Z). We try to address these challenges below.

Sufficiency term. Consider the first term in the objective function. We can write it as a function of the optimization parameters πθ(y|x,z),πθ(z|x) as follows:

πθ(y|x,z) P(y|x)

###### I(Y ;Z|X) =

P(x,y)P(z|x,y)log

x,y,z

πθ(y|x,z) P(y|x)

πθ(y|x,z) P(y|x)

P(x,y)πθ(z|x)

log

=

x,y,z

πθ(y|x,z) P(y|x)

≥

P(x,y)πθ(z|x)log

x,y,z

where we used xlog x ≥ log x in the last step. Therefore, we can maximize the lower bound and approximate it further using the query-answer samples (xi,yi). The first term of the optimization problem is:

m

θ(Z|xi)[log πθ(yi|xi,Z)]. (8)

EZ∼π

i=1

We can also approximate the bound using variational approximation of [3]. We introduce a verifier model Qρ(y|x,z) as variational parameter:

πθ(y|x,z) P(y|x)

###### I(Y ;Z|X) =

P(x,y)P(z|x,y)log

x,y,z

πθ(y|x,z)Qρ(y|z,x) P(y|x)Qρ(y|z,x)

P(x,y)P(z|x,y)log

=

x,y,z

Qρ(y|z,x) P(y|x)

+ E(X,Z)DKL(πθ(·|X,Z)∥Qρ(·|X,Z))

P(x,y)P(z|x,y)log

=

x,y,z

Qρ(y|z,x) P(y|x)

≥

P(x,y)P(z|x,y)log

x,y,z

πθ(y|x,z) P(y|x)

Qρ(y|z,x) P(y|x)

P(x,y)πθ(z|x)

=

log

x,y,z

Throughout the paper, we assume that there is always a unique answer to each query. Using this assumption, we can lower bound the last step as follows:

Qρ(y|z,x) P(y|x) ≥

πθ(y|x,z) P(y|x)

P(x,y)πθ(z|x)

log

x,y,z

P(x,y)πθ(z|x)log Qρ(y|z,x),

x,y=ytrue(x),z

where we used the assumption that P(y|x) = δ(y−ytrue(x)). Finally, we can maximize the following objective function for the sufficiency term:

m

EZ∼π

θ(Z|xi)[log Qρ(yi|xi,Z)].

i=1

Minimality term. For the minimality term, we use a variational approximation to find a variational bound similar to [3]. This upper bound combined with the above lower bound provides a general lower bound on the conditional information bottleneck objective which we try to maximize. First note that:

πθ(z|x) P(z)

P(x)πθ(z|x)log

###### I(X;Z) =

.

x,z

There is a dependence on P(z) which requires marginalization over x and is intractable. The derivation of the variational lower bound is quite standard:

πθ(z|x) P(z)

P(x)πθ(z|x)log

###### I(X;Z) =

x,z

πθ(z|x)

Qϕ(z) − DKL(P(Z)∥Qϕ(Z)) ≤

P(x)πθ(z|x)log

=

x,z

πθ(z|x) Qϕ(z)

P(x)πθ(z|x)log

.

x,z

The variational distribution Qϕ(·) is supposed to capture the distribution over reasoning traces without conditioning on the prompt.

The final optimization problem consists of finding a policy πθ(z|x) that maximizes the returns defined from the above approximate bounds using Qϕ and Qρ.

Marginal probability constraint. Let’s consider again the constraint for the conditional information bottleneck:

P(y|x) =

πθ(y|x,z)πθ(z|x).

z

As we mentioned above, the conditional probability is unknown. Now, assume that for each query there is a unique answer: P(y|x) = δ(y − ytrue(x)). In this case, the marginal probability constraint holds only if πθ(y|x,z) is also equal to δ(y − ytrue(x)), namely:

πθ(ytrue(x)|x,z) = 1, (9) We do not need explicitly include this constraint in the optimization problem, because it amounts to maximizing the probability of the correct answer under πθ(y|x,z) which is already part of the optimization problem.

#### B Additional Results

###### B.1 Domain Generalization

Tables 3–5 extend the summary in Table 1 by reporting per-benchmark standard error of the mean (SEM) compute over 10 seeds for all benchmarks expect for MATH500 and LiceCodeBench. We highlight several observations that complement the main-text analysis.

Statistical reliability. On competition-level math benchmarks (AIME24, AIME25), individual CIBβ− accuracy gains are typically comparable in magnitude to the SEM (e.g., DeepScaleR AIME24: 38.2±1.5 → 40.2±1.9; DLER-1.5B AIME24: 33.4±1.4 → 35.4±2.0; Table 3), which is expected given the inherent variance of 30-problem test sets. The meaningful signal is consistency: CIBβ− raises accuracy on 25 of 32 model×benchmark evaluations across Tables 3–5 In contrast, the L1-based degradations are individually unambiguous—L3L1-EXACT on AIME24 drops by 13.8 points (38.2±1.5 →24.4±1.3), DLER-7B L1-Exact on OlympiadBench by 3.1 points (65.0±0.4 → 61.9±0.7)—well beyond overlapping confidence intervals.

- Table 3: Accuracy (↑) and average CoT length (↓) — Math Benchmarks. Bold indicates best performance within each model group. red marks significant degradation. CIB subscripts denote the compression setting (β−=5 × 10−5, β+=1.5 × 10−4) and the prior size (1.5B or 7B). SEM is shown as subscript. ∗: Pass@1 with n = 16 for consistency, the official HuggingFace page reports pass@1(Avg@64); †: Our models; ‡: Models we trained using the L1-EXACT length penalty [1].

Model MATH500 AIME24 AIME25

Acc./Tok. Acc./Tok. Acc./Tok. DeepScaleR-1.5B

- Baseline 85.8±1.6 /3190±132 38.2±1.5 /9331±116 25.4±1.5 /8907±144 L3L1-EXACT 85.2±1.6 /1932±60 24.4±1.3 /2637±15 20.2±1.3 /2608±17

CIB†β

−,1.5B 87.6±1.5 /2359±121 40.2±1.9 /7298±141 27.8±2.1 /6866±170 CIB†β

+,1.5B 87.2±1.5 /2017±107 38.2±2.0 /6937±141 27.6±2.1 /6757±184 CIB†7B 86.1±1.6 /1337±93 37.5±1.9 /5003±143 25.0±1.9 /5389±164 Nemotron-1.5B

Baseline∗ 85.8±1.6 /8509±382 43.1±1.2 /22931±326 35.1±1.3 /24086±333

- L1-Exact‡ 85.8±1.5 /8807±393 40.2±1.4 /22079±383 30.8±1.4 /23243±400

CIB†β

−,1.5B 86.8±1.6 /7536±412 45.3±1.9 /21962±483 34.1±1.9 /22980±547 CIB†β

+,1.5B 85.4±1.6 /5982±354 41.5±1.9 /20954±454 34.0±1.8 /21407±542 CIB†7B 86.7±1.5 /5789±366 42.7±1.8 /20547±470 32.3±1.8 /20678±532 DLER-1.5B

Baseline 86.4±1.5 /1873±91 33.4±1.4 /3478±25 23.9±1.5 /3290±29

- L1-Exact‡ 86.6±1.5 /1751±111 35.0±2.1 /3274±34 22.7±2.0 /3037±41

CIB†β

−,1.5B 88.4±1.5 /1687±102 35.4±2.0 /3198±40 25.3±2.1 /3260±56 CIB†β

+,1.5B 87.6±1.4 /1421±64 34.1±2.0 /3124±43 24.2±2.1 /3027±47 CIB†7B 85.0±1.6 /805±35 31.6±2.0 /2504±37 22.4±1.9 /2257±38 DLER-7B

Baseline 93.6±1.1 /1159±41 49.6±1.7 /3221±47 36.6±1.6 /3271±54 L1-Exact‡ 92.8±1.2 /675±33 46.2±2.5 /2982±62 31.2±2.2 /2564±61 CIB†β

−,7B 94.0±1.1 /1033±39 49.7±2.3 /2642±69 37.1±2.2 /2962±77 CIB†β

+,7B 92.2±1.2 /678±34 48.2±2.2 /2419±75 35.4±2.3 /2550±72

Domain generalization. Table 5 confirms that CIB generalizes beyond formal mathematical reasoning despite being trained exclusively on math data. On DeepScaleR, CIBβ− improves GPQA Diamond by +1.9 (36.5±0.7→38.4±0.6) and MMLU-Pro by +2.2 (39.4±0.2→41.6±0.2), both well outside overlapping error bars. On DLER-7B, CIBβ− matches or slightly improves all three out-ofdomain benchmarks (GPQA 50.1→50.5; MMLU-Pro 58.7, unchanged; LCB 35.9→35.6) while reducing tokens by 10.9%–17.4%. LiveCodeBench accuracy remains within ≤1 point of the baseline for the β− and β+ settings across all model families; only the most aggressive 7B-prior configurations occasionally exceed this margin (DeepScaleR −1.3; DLER-1.5B −1.1). Two model-specific patterns are also worth noting. On DLER-1.5B, the baseline holds the highest GPQA Diamond accuracy in its group (40.9±0.7) and all compression methods reduce it, but CIBβ− degrades by only 1.5 points

- Table 4: Accuracy (↑) and average CoT length (↓) — Math Benchmarks. Bold indicates best performance within each model group. red marks significant degradation. CIB subscripts denote the compression setting (β−=5 × 10−5, β+=1.5 × 10−4) and the prior size (1.5B or 7B). SEM is shown as subscript. ∗: Pass@1 with n = 16 for consistency, the official HuggingFace page reports pass@1(Avg@64); †: Our models; ‡: Models we trained using the L1-EXACT length penalty [1].

Model Minerva Olympiad

Acc./Tok. Acc./Tok. DeepScaleR-1.5B

- Baseline 19.9±0.8 /6386±95 54.8±0.4 /5922±29 L3L1-EXACT 19.8±0.8 /2124±29 48.8±0.3 /2369±6

CIB†β

−,1.5B 20.4±0.8 /4510±77 55.5±0.7 /4472±47 CIB†β

+,1.5B 20.1±0.8 /4147±73 55.3±0.7 /4211±45

- CIB†7B 19.7±0.8 /2058±68 55.0±0.7 /3065±41 Nemotron-1.5B

Baseline∗ 11.0±0.5 /13011±191 43.8±0.3 /16270±89 L1-Exact‡ 11.4±0.4 /12925±147 44.0±0.4 /15316±127 CIB†β

−,1.5B 11.7±0.6 /11118±189 52.7±0.6 /15089±176 CIB†β

+,1.5B 10.9±0.6 /10051±168 50.8±0.6 /13894±173 CIB†7B 10.6±0.5 /8752±177 51.1±0.6 /12242±170 DLER-1.5B

Baseline 20.9±0.8 /2587±29 54.1±0.4 /2766±10 L1-Exact‡ 21.2±0.8 /2277±26 54.0±0.7 /2577±20 CIB†β

−,1.5B 21.6±0.8 /2346±27 54.3±0.7 /2616±18 CIB†β

+,1.5B 21.0±0.8 /2305±27 54.1±0.7 /2544±17

- CIB†7B 20.4±0.9 /1327±23 51.6±0.7 /1631±16 DLER-7B

Baseline 27.3±1.0 /1849±30 65.0±0.4 /2840±27 L1-Exact‡ 26.1±1.0 /1144±30 61.9±0.7 /1872±26 CIB†β

−,7B 27.6±1.0 /1381±26 64.1±0.7 /2361±39 CIB†β

+,7B 25.8±1.0 /851±19 62.7±0.7 /1704±28

(39.4±0.6), whereas L1-Exact drops by 4.1 points (36.8±0.6), indicating that CIB better preserves domain-specific knowledge even when some regression is unavoidable. On Nemotron, the 1.5B-prior CIB variants reduce MMLU-Pro accuracy by up to 2.0 points (41.4→39.4), but the 7B prior fully recovers it (41.6, +0.2 vs. baseline) while still compressing tokens by 19.0% (∼ 10.5K→ ∼ 8.5K), suggesting that the stronger prior compensates for the capacity gap on knowledge-intensive tasks.

- Table 5: Accuracy (↑) and average CoT length (↓) — Science, Knowledge & Code Benchmarks. Bold indicates best performance within each model group. red marks significant degradation. CIB subscripts denote the compression setting (β−=5×10−5, β+=1.5×10−4) and the prior size (1.5B or 7B). SEM is shown as subscript. ∗: Pass@1 with n = 16 for consistency, the official HuggingFace page reports pass@1(Avg@64); †: Our models; ‡: Models we trained using the L1-EXACT length penalty [1].

Model GPQA Diamond MMLU-Pro LCB

Acc./Tok. Acc./Tok. Acc./Tok. DeepScaleR-1.5B

Baseline 36.5±0.7 /7445±47 39.4±0.2 /5285±16 18.7±2.6 /11403±301 L3L1-EXACT 37.9±0.7 /3125±16 40.4±0.2 /2240±5 17.5±2.5 /4612±89

CIB†β

−,1.5B 38.4±0.6 /4571±98 41.6±0.2 /3286±14 18.8±1.2 /10410±151 CIB†β

+,1.5B 35.5±0.6 /4429±87 40.7±0.2 /3108±14 18.6±1.1 /10274±134 CIB†7B 39.5±0.6 /2850±87 40.8±0.2 /2118±13 17.4±1.1 /9481±126 Nemotron-1.5B

Baseline∗ 31.4±0.7 /21238±154 41.4±0.2 /10490±39 26.6±2.5 /24082±620 L1-Exact‡ 33.9±1.1 /21132±219 39.6±0.1 /10416±30 26.8±1.3 /23995±310 CIB†β

−,1.5B 34.9±0.6 /22558±299 39.4±0.2 /9851±41 26.1±1.2 /23557±295 CIB†β

+,1.5B 34.8±0.6 /19694±307 39.6±0.2 /8908±38 26.1±1.2 /22227±308 CIB†7B 33.6±0.7 /19533±319 41.6±0.2 /8496±39 25.9±1.2 /21817±304 DLER-1.5B

Baseline 40.9±0.7 /3280±16 41.7±0.2 /2575±12 17.9±2.5 /4857±89 L1-Exact‡ 36.8±0.6 /3012±30 40.8±0.2 /2326±11 17.8±1.1 /4585±39 CIB†β

−,1.5B 39.4±0.6 /3183±47 41.8±0.2 /2469±13 18.2±1.3 /4727±46 CIB†β

+,1.5B 38.8±0.5 /3044±36 41.6±0.2 /2353±12 17.9±1.1 /4629±42 CIB†7B 36.7±0.6 /2423±43 40.4±0.2 /1973±15 16.8±1.1 /3977±40 DLER-7B

Baseline 50.1±0.8 /3255±19 58.7±0.2 /2012±5 35.9±3.2 /4847±130 L1-Exact‡ 48.6±0.7 /2551±40 56.9±0.2 /1446±6 34.4±1.4 /4120±52 CIB†β

−,7B 50.5±0.8 /2899±33 58.7±0.2 /1718±5 35.6±1.4 /4003±55 CIB†β

+,7B 49.7±0.7 /2476±32 58.2±0.2 /1430±6 34.5±1.4 /3796±61

###### B.2 Scaling Properties

Figure 4– 7 show the Inference Time Compute (ITC) behavior of our CIB models compared to the baselines on two math reasoning benchmarks, namely, AIME24 and AIME25. Notably, CIBcompressed models exhibit on par or better scaling behavior than baselines. Especially, when bounding the maximum generation length to 3K, Figure 5, or 2K, Figure 6, we see that CIB trained model with the large prior (Qϕ = 7B) achieves superior scaling performance.

CIB Qϕ =7B (ours)

DLER-1.5B

CIB Qϕ =1.5B (ours)

L3L1-1.5B-EXACT

L3L1-1.5B-MAX

AIME24

AIME25

0.8

0.6

Accuracy(Pass@k)

0.7

0.5

0.6

0.5

0.4

0.4

0.3

0.3

0.2

0.2

Pass@1Pass@4Pass@16Pass@64Pass@128

Pass@1Pass@4Pass@16Pass@64Pass@128

- Figure 4: DLER — Inference Time Compute. Pass@k accuracy for different values of k, with a maximum completion length of 8K tokens.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Pass@1Pass@4Pass@16Pass@64Pass@128

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Accuracy(Pass@k)

AIME24

Pass@1Pass@4Pass@16Pass@64Pass@128

0.1

0.2

0.3

0.4

0.5

0.6

AIME25

DLER-1.5B

L3L1-1.5B-EXACT

L3L1-1.5B-MAX

CIB Qϕ =7B (ours)

CIB Qϕ =1.5B (ours)

- Figure 5: DLER — Inference Time Compute. Pass@k accuracy for different values of k, with a maximum completion length of 3K tokens.

CIB Qϕ =7B (ours)

DLER-1.5B

CIB Qϕ =1.5B (ours)

L3L1-1.5B-EXACT

L3L1-1.5B-MAX

AIME24

AIME25

0.45

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
| | | | | |
| | | | | |
| | | | | |

0.5

0.40

Accuracy(Pass@k)

0.35

0.4

0.30

0.3

0.25

0.20

0.2

0.15

0.1

0.10

Pass@1Pass@4Pass@16Pass@64Pass@128

Pass@1Pass@4Pass@16Pass@64Pass@128

- Figure 6: DLER — Inference Time Compute. Pass@k accuracy for different values of k, with a maximum completion length of 2K tokens.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Pass@1Pass@4Pass@16Pass@64Pass@128

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

Accuracy(Pass@k)

AIME24

Pass@1Pass@4Pass@16Pass@64Pass@128

0.1

0.2

0.3

0.4

0.5

0.6

0.7

AIME25

Deepscaler

L3L1-1.5B-EXACT

L3L1-1.5B-MAX

CIB Qϕ =1.5B (ours)

CIB Qϕ =7B (ours)

- Figure 7: DeepScaler — Inference Time Compute. Pass@k accuracy for different values of k, with a maximum completion length of 8K tokens.

#### C CoT Qualitative Comparison: Pruning Reasoning Redundancy

To examine the nature of the compression induced by our semantic prior, we visualize qualitative differences between baseline traces and CIB-generated traces across a range of reasoning tasks (see Figures 8–10). We find that the semantic prior does not merely truncate outputs; instead, it changes which computations are expressed in the trace. Concretely, by imposing an information-cost on the reasoning tokens (via the prior surprisal) while preserving task success through the sufficiency objective, CIB penalizes computation that offers low marginal information regarding the target Y . This effect manifests through three recurring mechanisms:

- • Induction of Algorithmic Generalization. Perhaps most notably, the information bottleneck biases the model toward theoretically superior solution paths. In geometric reasoning (Figure 8), while the baseline defaults to brute-force coordinate calculations via the Pythagorean theorem, the CIB model converges on a concise trigonometric identity (sinT = cosR). This suggests that minimizing the redundant computation under the semantic prior of the reasoning trace naturally selects for abstract, elegant proofs, as these represent the most compressed description of the transformation from prompt X to answer Y .
- • Suppression of Stochastic Exploration and Verification Bloat. Baseline models typically adopt a high-entropy strategy characterized by reasoning redundancy, utilizing conversational scaffolding and unstructured exploration. For instance, in arithmetic search tasks (Figure 9), the baseline explicitly calculates invalid candidates (e.g.,

983) before testing the correct one. Similarly, in constraint satisfaction problems (Figure 10), the baseline engages in tautological self-checks (e.g., verifying that positive lengths satisfy x > 0). CIB eliminates these low-utility branches. By penalizing the cumulative surprisal of the chain, the policy shifts from “exploratory thinking" to “efficient execution," treating valid derivations as terminal states rather than triggering redundant self-doubt loops.

- • Semantic Filtering of Syntactic Artifacts. CIB acts as a semantic filter that separates essential state information from syntactic artifacts. As shown in Figure 8, when presented with raw code metadata (Asymptote), the baseline expends significant budget on “verbal parsing"—reading the code aloud without progressing the state. The compressed policy bypasses this verbalization, extracting the underlying geometric conditions directly.

Problem: In △RST, shown, sinR = 25. What is sinT? [Asymptote Code Segment]: pair R,S,T; T = (0,0); S = (2,0); R =

(2,sqrt(21)); ...

###### Baseline Response (Standard CoT) - 1685 tokens

Okay, so I have this triangle RST... The Asymptote code draws a right-angled triangle... The Asymptote code defines points T at (0,0)... Triangle RST is right-angled at S. So, angle at S is 90 degrees... Wait, the label is on (R+T)/2... RT is the hypotenuse, length 5. Let me assign the sides... Given sin R = 2/5... sin R = ST / RT = 2/5. RT is 5, so ST = 2. Now, since it’s a right-angled triangle at S, we can use Pythagoras... RS2 + 22 = 52... RS = √21. Now, we need to find sin T... Wait, no. Wait, angle T is at point T... Wait, maybe I... sin T = opposite / hypotenuse... RS / RT. sin T = √21/5.

###### Ours: CIB - 154 tokens

First, observe that triangle RST is a right-angled triangle at point S. Given that sin R = 2/5. In a right-angled triangle, the sum of angles R and T is 90 degrees, so angle T is complementary to angle R. Using the sine of complementary angles, sin T = cos R.

Since sin R = 2/5, cos R = 1 − (2/5)2 = 21/25 = √21/5. Therefore, sin T = √21/5.

- Figure 8: Qualitative comparison on geometry reasoning. Top: Prompt. Middle: the Baseline trace is dominated by redundant “verbal parsing" of the input code and repetitive self-correction loops (highlighted in red). Bottom: The CIB trace successfully filters this syntactic noise. Notably, the information constraint induces a shift in strategy: CIB bypasses the lengthy coordinate calculation favored by the baseline, converging instead on a concise trigonometric identity.

Problem: When the two-digit integer “AB” is cubed, the value is 912,673. What is A + B?

Baseline Response (Standard CoT) - 965 tokens Okay, so I have this problem... I need to find A + B... Hmm... Wait, let me think. AB is a two-digit integer... between 10 and 99. First, maybe I should find the cube root... I know that 1003 is 1,000,000... Maybe 993 is about 970299... 983 is 98 * 98 * 98. Let me calculate that. 9604 * 98... 960400 - 19208 = 941192. So 983 is 941192, which is larger than 912673. What about 973? Let me compute that. 97 * 97 is 9409... 9409 * 97 = 912,673. Oh, wait! 97 cubed is 912673. So AB is 97. Wait wait, let me double-check. 97 cubed is 912673... Yes, correct. So AB is 97, so A is 9, B is 7. Therefore, A + B is 9 + 7 = 16.

Ours: CIB - 210 tokens To determine the value of A + B, we start by noting that the two-digit integer AB is a number between 10 and 99. When AB is cubed, the result is 912,673. Taking the cube root of 912,673 gives an approximate value. The cube root of 912,673 is approximately 97, since 973 = 912,673. Thus, AB is 97. The digits A and B are 9 and 7, respectively. Adding them together, A + B equals 9 + 7 = 16.

- Figure 9: Qualitative comparison on arithmetic search. Top: Prompt. Middle: Baseline model engages in inefficient trial-and-error, explicitly calculating the incorrect candidate 983 (highlighted in red) and engaging in redundant self-verification loops. Bottom: the CIB model (bottom) suppresses this exploratory computation, converging directly on the correct candidate (97) and reducing the token count by ∼78% without loss of accuracy.

Problem: Two sides of a triangle are each 8 units long. If the third side has a whole number length, what is the greatest possible perimeter, in units, for the triangle?

Baseline Response (Standard CoT) Okay, so I have this problem... Hmm, let me think about how to approach this. Triangle

###### inequality theorem... sum of any two sides > third. 8 + 8 > c. That means 16 > c...c can be at most 15. But wait, wait. Let me think again... The triangle inequality also requires... 8 + c > 8. Well,

that simplifies to c > 0, which is always true... But wait, let me double-check. Let’s plug c = 15 into the triangle inequalities... 1. 8 + 8 > 15? Yes... 2. 8 + 15 > 8? Yes... So all the inequalities are satisfied. Wait, but hold on. Let me make sure I didn’t miss any constraints... c can be 15... Perimeter is 8 + 8 + 15 = 31 units.

###### Ours: CIB

To determine the greatest possible perimeter... we apply the triangle inequality theorem. The sum of any two sides must exceed the third side. Let the third side be ‘x’. The inequalities are: 1. 8 + 8 > x → x < 16 2. 8 + x > 8 → x > 0 (always true) Thus, x must be less than 16 and a positive integer. The maximum integer less than 16 is

###### 15. Therefore, the maximum perimeter is 8 + 8 + 15 = 31.

- Figure 10: Qualitative comparison on constraint satisfaction.Top: Prompt. Middle: Baseline trace is characterized by “verification bloat." Despite correctly deriving the constraint (c < 16) early on, the model expends tokens checking tautologies (8 + c > 8) and re-verifying its own conclusions (highlighted in red). Bottom: CIB trace (bottom) retains the necessary constraint logic but eliminates the redundant self-auditing loops, trusting the derivation immediately.

#### D Training Details

To ensure reproducibility, we provide the full set of hyperparameters and infrastructure details used in our experiments. Our implementation relies on the trl library (version 0.26.2) for Group Relative Policy Optimization (GRPO) and lighteval (version 0.8.1) for robust evaluation.

- D.1 Hyperparameters We fine-tuned all models using the hyperparameters listed in Table 6.

Table 6: GRPO Training Hyperparameters. All experiments share these settings unless otherwise noted.

Hyperparameter Value Optimizer AdamW Learning Rate 1 × 10−6 LR Scheduler constant Warmup Ratio 0.03 Batch Size (Global) 128 Number of Generations per Prompt 16 Temperature 0.8 Max Completion Length 4096 for DLER and 8192 for Deepscaler Max Gradient Norm 1.0 KL Penalty Coefficient (βKL) 5.e−4 CIB Regularization Weight (β−,β+) {5.e−5,1.5e−4} Number of steps 150

- D.2 Evaluation Setup We utilized lighteval for all downstream benchmarks.

- • Inference Engine: vLLM (version 0.10.2).
- • Sampling Strategy: We used temperature T = 0.6, topp = 0.95, 32K max completion length, and 16 generations per prompt.
- • Hardware: Training was performed on a node with 8× NVIDIA H100 (80GB) GPUs.

##### E Results from literature We report additional results from [46] in Table 7.

Table 7: Compression results for DeepScaler-1.5B [46]. Model MATH500 AIME24 AMC-23 OlympiadBench

Acc.↑ Tok.↓ Acc.↑ Tok.↓ Acc.↑ Tok.↓ Acc.↑ Tok.↓ DeepScaler-1.5B 85.8 3280 35.5 9246 74.2 6416 54.6 5974 ThinkPrune-I2k 85.5 1707 34.9 5095 74.3 2913 54.7 3498 ThinkPrune-4k 86.6 2042 35.5 6488 76.3 3839 55.7 4010 HAPO 84.4 2370 31.4 7702 70.3 4301 51.4 4571 AutoThink 84.9 1635 36.2 7201 67.8 3658 52.5 4085 Thinkless 81.3 2944 28.9 9143 65.7 5276 50.2 6057 LAPO-D 86.4 2365 37.6 5945 77.6 3655 56.1 4499 LAPO-I 86.3 2168 38.1 5371 78.3 3765 56.3 4024

#### F The list of datasets and models

- Table 8: The list of datasets.

Dataset Link License MATH500 [21] HuggingFace MIT

- AIME 2024 [28] HuggingFace Copyright MAA
- AIME 2025 [29] HuggingFace Copyright MAA Minerva Math [17] HuggingFace MIT OlympiadBench [9] HuggingFace MIT AMC 2023 [27] HuggingFace Copyright MAA GPQA-Diamond [32] HuggingFace CC BY 4.0 MMLU-Pro [43] HuggingFace MIT LiveCodeBench [13] GitHub MIT DeepScaleR dataset [24] HuggingFace MIT

- Table 9: The list of models.

###### Model Link License

DLER-R1-1.5B [22] HuggingFace NSCLv1 (research only) DLER-R1-7B [22] HuggingFace NSCLv1 (research only) DeepScaleR-1.5B-Preview [24] HuggingFace MIT

- Qwen2.5-Base-0.5B [50] HuggingFace Apache 2.0
- Qwen2.5-Base-1.5B [50] HuggingFace Apache 2.0 Qwen2.5-Base-3B [50] HuggingFace Qwen Research Qwen2.5-Base-7B [50] HuggingFace Apache 2.0 L3L1-1.5B-Exact [1] HuggingFace MIT OpenReasoning-Nemotron-1.5B [2, 30] HuggingFace CC-BY-4.0

