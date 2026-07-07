## How LoRA Remembers? A Parametric Memory Law for LLM Finetuning

Ziwen Xu1,2*, Haiwen Hong2*, Linsong Yu1*, Benglei Cui2, Longtao Huang2, Hui Xue2, Ningyu Zhang1† 1Zhejiang University, 2Alibaba Group

# arXiv:2605.30260v1[cs.CL]28May2026

Abstract

Large Language Models (LLMs) must continuously learn and update knowledge to remain effective in dynamic real-world environments. While Low-Rank Adaptation (LoRA) is widely used for such memory updates, existing studies mainly rely on qualitative downstream evaluations, leaving the quantitative capacity limits and underlying dynamics of exact parametric memory largely unexplored. To bridge this gap, we employ LoRA as a controlled memory capacity probe within the latent space to systematically quantify exact parametric memory. We introduce the Parametric Memory Law, a robust power law linking loss reduction ∆L to effective parameters and sequence length. At the token level, fine-grained analysis reveals a deterministic phase transition, demonstrating that a prediction probability of p > 0.5 constitutes a sufficient condition for verbatim recall under greedy decoding. Driven by these insights, we introduce MemFT, a threshold-guided optimization strategy that dynamically redistributes the training budget toward sub-threshold tokens. Empirical evaluations demonstrate that MemFT can enhance memory fidelity and efficiency1.

### 1 Introduction

Large Language Models (LLMs) have shown strong capabilities across diverse tasks and are now widely used in real-world systems (Zhao et al., 2023). However, their knowledge is encoded in static pretrained parameters and remains largely fixed after deployment. In practice, models continuously encounter new information such as updated facts, user preferences, and task-specific knowledge (Yao et al., 2023). Efficiently integrating such information therefore becomes an key problem in continual learning and memory systems.

* Equal Contribution. † Corresponding Author. 1Code will be released at https://github.com/zjunlp/

ParametricMemoryLaw.

[Figure 1]

Figure 1: LoRA as a pluggable memory unit in the LLM’s latent space. The LoRA module (rank r) encodes contextual knowledge into the residual stream at layer k, enabling faithful recall of memorized text. The Parametric Memory Law quantifies the capacityparameter trade-off.

Non-parametric methods address this challenge by providing external context during inference. Specifically, approaches such as in-context learning (ICL) (Dong et al., 2024), retrieval-augmented generation (RAG) (Gao et al., 2023), and external non-parametric memory systems (He et al., 2024; Fang et al., 2025) dynamically integrate information without modifying model parameters. However, these methods are fundamentally constrained by fixed context windows, attention dilution, and escalating computational overhead as the input length scales (Liu et al., 2024).

In contrast, parametric memory embeds information directly into parameters or modular structures, enabling permanent knowledge consolidation and retrieval-free internal reasoning (Yang et al., 2024; Li et al., 2025; Lei et al., 2026). Recent works have further conceptualizes Low-Rank Adaptation (LoRA) as a specialized knowledge memory unit (Back et al., 2026). However, existing evaluations predominantly rely on downstream functional tasks, such as question answering. While effective for demonstrating practical utility of LoRA and its synergy with non-parametric methods (Back et al., 2026; Su et al., 2025), these benchmarks inevitably conflate raw information memorization with downstream comprehension and instruction-following

behaviors. Consequently, the isolated quantitative capacity boundaries and dynamics mechanisms of parametric memory remain under-explored.

To bridge this gap, we focus on exact parametric memory. Drawing from fuzzy-trace theory in cognitive science (Reyna and Brainerd, 1995), memory dual-encodes information into independent gist and verbatim traces. While functional benchmarks evaluate gist-level capability, exact text reconstruction isolates verbatim retention. Crucially, while nonparametric approaches guarantee verbatim output by directly fetching source text, parametric memory lacks this advantage and struggles with faithful reconstruction. Characterizing exact memory within parametric structures is therefore foundational. Using LoRA as a parameter-controllable probe within the latent space, we investigate:

|Research Question|
|---|

What governing principles dictate the capacity boundaries and dynamic mechanisms of exact parametric memory?

As illustrated in Figure 1, we investigate exact parametric memory by scanning rank and memory sequence configurations. Globally, the LoRAinduced loss reduction ∆L follows a stable powerlaw scaling with effective parameters and sequence length, which we formalize as the Parametric Memory Law. At the token level, however, finegrained analysis reveals demonstrate that a low average loss does not guarantee memorization. Specifically, under greedy decoding, a token prediction probability of p > 0.5 is a sufficient condition to lock it into a stable state. Below this threshold, stubborn tokens face high-entropy competition with alternative tokens, sharply increasing the risk of autoregressive cascade failure. Based on these insights, we propose MemFT, an optimization strategy that redirects the parameter budget to sub-threshold tokens to maximize efficiency.

Our main contributions are:

- • Parametric Memory Law: We establish a power law that quantifies exact memory capacity based on parameters and sequence length.
- • Dynamics Mechanism: We reveal that low average loss hides token-level competition, identifying p > 0.5 as a sufficient condition for memory locking and lower probabilities as a catalyst for cascade collapse.

• MemFT Method: We develop a thresholdguided optimization targeting stubborn tokens, significantly surpassing standard SFT in both fidelity and parameter efficiency.

### 2 Preliminary

2.1 Exact Parametric Memory Task Task Setup. Inspired by Jelassi et al. (2024); Back et al. (2026), we formulate exact parametric memorization over a dataset D = {(q(i),a(i))}Ni=1, where q(i) serves as a unique key and a(i) = (a(1i),...,a(ℓi)

) is the target content. Given a frozen

i

base model fθ0, we learn a parameter increment ∆θ to construct an updated model fθ with parameters θ = (θ0,∆θ), satisfying:

fθ(q(i)) = a(i), ∀i ∈ {1,...,N}. (1)

Since a(i) is inaccessible during inference except via the query q(i), ∆θ constitutes the exclusive medium for information storage. This reduces memorization to pure parameter writing, decoupling it from retrieval or contextual comprehension.

Answer-only Accounting. Since questions serve only as keys, all token-level quantities in this paper (sequence length ℓ, loss, accuracy) are computed exclusively over answer tokens; question tokens provide conditioning context and are excluded from every reported metric. Notably, the sequence length ℓ is determined by tokenizing the answer using each model’s respective tokenizer.

Evaluation Metrics. Exact memorization demands a deterministic, reproducible decoding rule; thus, we adopt greedy decoding (aˆt = arg maxv∈V pθ(v | q,a<t)) throughout this work. We monitor three standard metrics to capture model behavior at different granularities.

Let a = (a1,...,aℓ) be the ground-truth answer of length ℓ, and Lt(θ) = −log pθ(at | q,a<t) be the token-level cross-entropy loss.

Sequence-Averaged Loss. The macroscopic loss serves as a global optimization proxy:

1 ℓ

L(a;θ) =

ℓ

Lt(θ). (2)

t=1

Drawing on the view of language modeling as compression (Delétang et al., 2024), we treat L(a;θ) as a measure of memorization for sequence a, where the loss reduction ∆L quantifies the memory gain.

Token-Level Accuracy. This metric measures the fraction of correctly predicted tokens, providing a microscopic view of memorization progress:

1 ℓ

Acctok(a;θ) =

ℓ

1[ˆat = at]. (3)

t=1

Exact Match Accuracy. This strict binary metric evaluates whether the entire sequence is reproduced verbatim:

AccEM(a;θ) = 1[∀t ∈ {1,...,ℓ},aˆt = at]. (4)

We observe that L and Acc are not monotonically aligned. Consequently, we track all three metrics: L for global convergence trends, Acctok for fine-grained token-level dynamics, and AccEM for strict recall fidelity.

- 2.2 LoRA-based Parametric Memory Injection

We realize ∆θ with LoRA: for each adapted linear layer with frozen weight W0 ∈ Rdout×din, we attach a low-rank residual branch so that its forward pass becomes

h = W0x + BAx,A ∈ Rr×din, B ∈ Rdout×r,

(5) where r ≪ min(din,dout) is the LoRA rank and ∆θ collects all such {(Aℓ,Bℓ)}ℓ. We view LoRA as a latent-space probe: r is a single monotone knob on the trainable parameter count, letting us sweep the capacity axis and cleanly examine how parameter budget relates to memory capacity. Since θ0 is frozen, any change in L or Acc is attributable solely to ∆θ. At inference time ∆θ is used through the residual branch in Eq. 5; whether BA is fused back into W0 is an implementation choice that leaves fθ unchanged.

- 3 The Parametric Memory Law

In this section, we discover the Parametric Memory Law through large-scale quantitative experiments, which governs the macroscopic scaling behavior of parametric memory in LLMs.

- 3.1 Empirical Observation: Linearity in Log-Log Space

To investigate the scaling dynamics of parametric memory, we define Loss Reduction as ∆L = Linit − Lfinal, where Linit and Lfinal denote the cross-entropy losses before and after applying parametric memory, respectively.

We conducted experiments on Qwen3-8BIT (Team, 2025) and Llama3.1-8B-IT (Team, 2024). The experimental design covered two typical scenarios: (1) a Long-Context Memorization Stress Test, inspired by (Zhu et al., 2024), using a LongBench (Bai et al., 2024) sample with 0%-100% token replacement by randomly sampled Qwen vocabulary tokens to generate different levels of semantic coherence and difficulty; (2) a Short-Context Dense Memory Test using PhoneBook datasets (Jelassi et al., 2024; Back et al., 2026) to evaluate high-density storage limits. We varied LoRA ranks r and sequence lengths ℓ extensively across these settings.

We analyze the relationship among ∆L, r, and ℓ across a wide range of experimental settings. As illustrated in Figure 2a, we observe distinct linear trends in the log-log domain. Specifically, ∆L scales positively with rank r and negatively with length ℓ. This high degree of linearity strongly suggests an underlying power-law relationship between ∆L and the parameters (r,ℓ).

Empirically, we exclude saturated samples with

Lfinal ≤ 0.69, with the threshold’s origin detailed in Section 4.3.

#### 3.2 Formulating the Parametric Memory Law

Based on the observed log-linearities, we formalize the scaling behavior as the Parametric Memory Law and propose the following empirical model:

∆L(r,ℓ) = C · rα · ℓ−β + b (6)

Here, ∆L denotes the training memory gain, while C is a scaling constant dictated by model and data distribution. α (Capacity Exponent) quantifies the efficiency of parameter rank in enhancing memory capacity. β (Length Penalty Exponent) reflects the nonlinear increase in memory difficulty associated with longer sequences. C,α,β are positive.

This law indicates that within the significant memory gain regime, performance is governed by a power-law trade-off between rank and length.

#### 3.3 Fitting Validation

We validated the Parametric Memory Law (Eq. 6) against the experimental data from Section 3.1, reporting both the coefficient of determination (R2) and Mean Absolute Percentage Error (MAPE) to assess goodness-of-fit.

As shown in Table 1, the law demonstrates exceptional explanatory power across diverse

- 100

- 101

- 100

- 101

Length

Rank r

L∆

L∆

L∆

200 500

- 5k

- 6k

- 7k

- 8k

- 1

- 2

10 12 14 16 24

- 1k

- 2k

- 3k

- 4k

4 6 8

10k

Rank r

Length

103 104

100 Rank r 101

Length

(a) Approximate log-linear dependence of ∆L on rank r and length ℓ

1.0

[Figure 2]

[Figure 3]

512 256 128

[Figure 4]

[Figure 5]

512 256 128

10

R2 = 0.996

- 100

- 101

0.8

64 48 32 24 16 14 12 10

64 48 32 24 16 14 12 10

PredictedL∆

8

FinalLoss

Accuracy

Rankr

Rankr

0.6

6

0.4

4

8 6 4 2 1

8 6 4 2 1

0.2

2

∆L(r, ) = C · rα · −β + b

0.0

501002005001k2k3k4k5k6k7k8k10k

501002005001k2k3k4k5k6k7k8k10k

100 True ∆L 101

Length

Length

(b) Predicted vs. true ∆L

(c) Decoupling of low loss and high accuracy

- Figure 2: Empirical validation of the Parametric Memory Capacity Law (LoRA on Qwen3-8B). (a) ∆L exhibits approximate log-linear decay with respect to rank r and length ℓ, forming a nearly planar structure in log-space; (b) The scatter plot compares predicted ∆L from Eq. (6) against true values, showing high fidelity (R2 = 0.996); (c) Heatmaps plot the final loss and token-level accuracy (correct tokens / total length) across various (r,ℓ) settings, revealing numerous cases where loss approaches zero while accuracy remains near zero.

models and data distributions. Specifically, it achieves R2 > 0.98 with low MAPE in all settings, including pure semantic, fully random, and shortcontext PhoneBook tasks. Notably, the law exhibits strong robustness to varying semantic densities. A single unified formula accurately fits the entire Long-context mixture (0%–100% random), yielding high R2 values of 0.987 for Llama3.1-8B-IT and 0.983 for Qwen3-8B-IT. These results confirm that the power law precisely characterizes the scaling of loss reduction. This consistency holds regardless of whether the context is structured or random, spanning from long-context to short-context scenarios.

masks the discrete nature of token-level memory. This section reveals the misalignment between loss and accuracy and establishes the critical point of the deterministic phase transition that determines the success or failure of memory.

#### 4.1 The Loss-Accuracy Misalignment

In exact parametric memory tasks, minimizing average cross-entropy loss does not guarantee high token-level accuracy, a phenomenon we term the Loss-Accuracy Misalignment.

Figure 2c shows models achieving near-zero average loss yet negligible accuracy. This occurs because average loss smooths over local variations, allowing high confidence on easy tokens to mask catastrophic errors on hard ones. As illustrated in Figure 3a, specific positions maintain persistently low probabilities (p < 0.5) despite low global loss, creating invisible bottlenecks.

In summary, the Parametric Memory Law provides a robust macroscopic mapping between parameter budget, sequence length, and loss reduction. However, by focusing on aggregate metrics, it abstracts away the microscopic dynamics of individual token memorization, which we analyze in the next section.

In autoregressive generation, such local errors are fatal. A single misprediction alters the context for subsequent steps, triggering error propagation that collapses the sequence. Thus, average loss is an insufficient proxy for generation fidelity, necessitating a shift to token-level probability analysis.

### 4 The Deterministic Phase Transition of Memory

The parametric memory law describes the macroscopic scaling behavior, but the average loss metric

Long-context memorisation stress test Phonebook r0 r20 r40 r60 r80 r100 Comb. All

Model Metric

R2 ↑ 0.992 0.994 0.996 0.995 0.996 0.996 0.987 0.981 MAPE(%)↓ 1.430 2.493 2.528 2.755 2.710 2.563 7.057 1.606

Llama3.1-8B-IT

R2 ↑ 0.996 0.993 0.996 0.996 0.995 0.996 0.983 0.990 MAPE(%)↓ 0.752 2.553 2.331 2.862 3.944 3.472 8.320 0.476

Qwen3-8B-IT

- Table 1: Goodness-of-fit of the parametric-memory law ∆L(r,ℓ) = C · rα · ℓ−β + b (Eq. 6) on two parametricmemorization benchmarks. We report two complementary metrics: R2 (higher is better) and MAPE (lower is better). The long-context memorization stress test sweeps the random-token ratio (rx=x% random; r0=pure LongBench, r100=fully random), and Comb. pools the six mixtures. Phonebook stores many short (name→number) key–value pairs, probing the short-text memory regime complementary to long-context.

#### 4.2 Token-Level Probability Dynamics

To uncover the microscopic origin of the LossAccuracy Misalignment, we analyze the per-token probabilities p(ti | t<i) after SFT convergence.

We identify stubborn token positions as indices where target token probabilities remain persistently below the p = 0.5 threshold, regardless of LoRA rank increases (Figure 3a; full grids across data scenarios in Appendix G). These bottlenecks are highly localized; for instance, Figure 3c shows that position i = 153 alone accounts for 28% of all failures, indicating that these are intrinsic hard cases resistant to capacity scaling.

Crucially, these stubborn positions drive autoregressive collapse. As demonstrated in Figure 3b, the earliest stubborn position tightly bounds the first decoding failure i∗ (Spearman ρ = 0.908). When p < 0.5, the correct token loses probabilistic dominance and becomes highly susceptible to being superseded by incorrect candidates during greedy decoding. This triggers cascading failures, corrupting all subsequent tokens and explaining why a single local bottleneck leads to complete sequence failure.

#### 4.3 Deterministic Phase Transition

The analysis above directs our attention to the critical role of the p = 0.5 threshold. Under greedy decoding, this probability value serves as the boundary for deterministic memory success, leading us to define the Deterministic Phase Transition.

Greedy decoding selects the token with the highest predicted probability. For successful memory, the target token must be the most probable candidate. A sufficient condition to guarantee this dominance is Ptarget > 0.5, as no other single candidate can exceed this value if the target holds the major-

ity of the probability mass. Thus, Ptarget = 0.5 acts as the critical threshold for deterministic success.

This probability threshold corresponds to a critical loss value. Given the cross-entropy loss L = −log(Ptarget), substituting Ptarget = 0.5 yields:

Lcrit = −log(0.5) = ln(2) ≈ 0.693 (7)

This derivation provides the theoretical basis for the empirical threshold Lfinal > 0.69 in Section 3.1. We characterize the memory states relative to this boundary: (1) Disordered Phase (L > Lcrit): Here, Ptarget < 0.5. The correct token does not hold a dominant probability, making it susceptible to being outcompeted by other candidates, thus leading to potential memory failure. (2) Ordered Phase (L < Lcrit): Here, Ptarget > 0.5. The correct token is guaranteed to be the most probable candidate, ensuring successful reproduction under greedy decoding.

Thus, Lcrit represents a sharp phase transition boundary between uncertain and deterministic memory success. The Parametric Memory Law describes the scaling trend of loss reduction, while the Deterministic Phase Transition explains why loss must cross this barrier to translate into effective accuracy. Pursuing lower loss aims to increase the confidence margin, but the acquisition of reliable memory capability begins with crossing this deterministic phase transition.

### 5 MemFT: Methodology and Empirical Verification

#### 5.1 The MemFT Method

Standard SFT minimizes the token-averaged crossentropy, allocating equal gradient budget to all tokens regardless of their learning status. As established in Section 4.3, tokens with loss L < Lcrit

1.00

0.75

|p(tt)i<i

p = 0.5

0.50

0.25

r = 32 r = 64 r = 256

First-failure token at i = 153

0.00

0 1000 2000 3000

Token position i

(a) Probability dynamics across ranks

4.0

[Figure 6]

Spearmanρ = 0.908 (n=155)

First-failurepostion∗i

- 100

- 101

- 102

- 103

3.5

3.0

log

2.5

i∗ = posp<0.5

i∗ ≥ posp<0.5 2.0

0

0 100 101 102 103

Earliest position with p < 0.5

(b) Lower bound on first failure

50

i∗ = 153(count = 44)

i∗ = 86(count = 14) i∗ = 96(count = 13)

40

Count

30

20

10

0

0 250 500 750 1000 1250 1500 1750

First-failure postion i∗

(c) Localization of failure positions

- Figure 3: Microscopic origin of the Loss-Accuracy Paradox. Results are based on Qwen3-8B trained on the Random dataset. (a) Sparse stubborn positions: A small set of indices where target probabilities persistently remain p < 0.5, resisting improvement even as LoRA rank increases. (b) Correlation with decoding failure: The earliest stubborn position tightly bounds the first failure index i∗ (Spearman ρ = 0.908, n = 155), indicating that sub-threshold probabilities significantly increase the likelihood of errors due to lost probabilistic dominance. (c) Spatial concentration: The histogram of first-failure positions across all settings reveals high localization.

Method Long-Context Memorization Stress Test (Acctok %) PhoneBook (AccEM %)

r1 r2 r3 r4 r5 r6 r7 r8 r9 p1 p2 p3 p4 p5 p6 p7 Llama3.1-8B-IT

SFT 27.4 28.5 43.6 45.9 54.9 69.5 78.2 86.3 94.7 0.50 3.85 18.7 28.0 37.8 47.0 59.3 MemFT-OT 27.3 36.4 45.6 54.7 63.6 70.5 85.4 94.7 100.0 1.00 11.2 31.4 53.9 61.0 73.9 87.0 MemFT-SW 32.5 37.5 46.0 52.3 56.0 63.4 69.1 76.6 81.1 1.84 15.0 34.0 45.7 70.7 96.1 100.0

###### Qwen3-8B-IT

SFT 17.9 24.2 27.8 31.7 33.1 39.8 40.2 40.0 47.7 2.32 17.4 37.5 55.5 84.8 99.5 100.0 MemFT-OT 19.2 23.6 29.8 38.5 47.5 56.1 91.1 100.0 100.0 5.78 19.1 36.2 57.4 86.1 98.6 100.0 MemFT-SW 24.7 29.3 32.0 39.4 52.5 74.6 93.5 94.4 94.4 8.45 19.7 37.8 58.8 86.5 99.5 100.0

- Table 2: Performance evaluation in the Long-Context Memorization Stress Test and the PhoneBook benchmark. Bold indicates the top-performing method within each rank budget per model. Rank Mapping: For LLAMA3.1-8B-IT, the long-context rank configurations r1 ...r9 denote {1,2,4,6,8,10,12,14,16}, respectively. For QWEN3-8B-IT, r1 ...r9 map to {1,2,4,8,16,32,64,128,256}. The PhoneBook rank indices p1 ...p7 represent {1,2,4,8,16,32,64} uniformly across both models.

are already in the ordered phase and effectively memorized. Continuing to optimize these tokens dilutes the signal for stubborn tokens (those in the uncertain regime), which are critical for preventing autoregressive error propagation.

To address this, we propose Memorizationoriented Fine-Tuning (MemFT), which replaces the uniform objective with a token-weighted form:

wt Lt(θ) t∈M wt + ε

LMemFT(θ) = t∈M

, (8)

where M is the set of target token indices t in the sequence, Lt(θ) is the cross-entropy loss at position t, and wt is a dynamic weight. Normalizing by the sum of weights ensures stable gradient scales across samples with varying numbers of active tokens. Different instantiations of MemFT differ only in the construction of wt.

MemFT-OT: Only Threshold Variant. The baseline uses the critical loss as a hard mask:

wtTH = 1[Lt > Lcrit]. (9)

Gradients are concentrated exclusively on tokens that have not yet crossed the phase transition. This avoids over-optimization of easy tokens and introduces no additional hyper-parameters.

MemFT-SW: Adaptive Sliding Mechanisms. MemFT-SW extends MemFT-OT by introducing two sliding strategies operating at different granularities to optimize gradient flow, which can be applied independently or in combination.

Intra-sample Spatial Sliding. To mitigate local bottlenecks, this mechanism dynamically focuses optimization on the context of the first prediction error. We define the anchor position ai as the first token where the greedy prediction deviates from the ground truth, and employ an exponential decay

function ϕt = exp(−max(t − ai,0)/τ) to weight the surrounding tokens. The final weight modulates

the base soft-threshold weight wtbase = σ(κ(ℓt − Lcrit)) using a sliding window of length Lwin:

ϕt, t < ai + Lwin, ϵfloor, t ≥ ai + Lwin.

wtseq = wtbase ·

(10)

Here, Lwin is initialized to a base value L0. The decay ϕt ensures that tokens upstream of the anchor (t < ai, where ϕt = 1) retain their base weights, while downstream tokens within the window are prioritized based on proximity to ai. To prevent stagnation, Lwin expands proportionally if ai remains static, and resets once the anchor advances.

Inter-batch Temporal Curriculum. This mechanism controls the exposure to complex samples across training steps. Within each epoch, we restrict optimization to a sliding window of batches Bcur, determined by training progress γ ∈ [0,1]. Early in training, only the first fraction of batches (e.g., those containing simpler or shorter sequences) are processed; as γ increases, the window expands to include all batches. This prevents the model from being overwhelmed by global complexity before stabilizing local memorization. Detailed hyperparameters are listed in Appendix D.

#### 5.2 Experimental Setup

We evaluate performance on two complementary benchmarks. The Long-Context Memorization Stress Test probes pure parametric capacity by focusing on its maximal difficulty regime, which consists entirely of semantic-free random tokens to eliminate linguistic priors. The PhoneBook (Jelassi et al., 2024) benchmark assesses the precise memorization of discrete key-value pairs, such as name-to-number mappings, in a short-text setting. We provide dataset construction in Appendix A.

We fine-tune QWEN3-8B-IT and LLAMA3.18B-IT with LoRA across varying ranks r and lengths L, comparing SFT, MemFT-OT, and MemFT-SW. Details are provided in Appendix B. We report token-level matching accuracy (correct tokens / total tokens) for the Long-Context test and exact match accuracy for PhoneBook. This dualmetric approach aligns with our phase-transition analysis in Section 4.3 for long sequences while ensuring strict fidelity for short factual recall.

#### 5.3 Main Results

Table 2 evaluates MemFT variants against the SFT baseline across varying parameter capacities.

In the Long-Context Memorization Stress Test, we observe a distinct capacity-dependent regime shift. In low-rank configurations (r1 ...r3), MemFT-SW consistently outpaces MemFT-OT and SFT. As the rank expands, however, MemFT-OT exhibits sharper acceleration, achieving perfect memory saturation (100.0% Acc) at Llama-r9 and Qwen-r8, ultimately surpassing MemFT-SW in high-rank settings.

Conversely, in the PhoneBook benchmark, MemFT-SW maintains a stable lead across almost all budget scales. It is the fastest to reach 100.0% EM accuracy (p7 for Llama and p6 for Qwen), while standard SFT struggles to achieve perfect recall under lower parameter budgets.

Overall, both MemFT variants consistently outperform standard SFT, demonstrating that threshold-guided training effectively bridges the parameter utilization gap to achieve high-fidelity exact reconstruction.

Rank Method Memory (%) Generalization (%)

- 1

SFT 83.0 19.0 MemFT 95.0 34.0 ↑ 15.0

- 2

SFT 100.0 38.0 MemFT 97.0 47.0 ↑ 9.0 4

- SFT 99.0 46.0 MemFT 100.0 53.0 ↑ 7.0 8

- SFT 100.0 39.0

- MemFT 99.0 49.0 ↑ 10.0

16

SFT 100.0 41.0

- MemFT 100.0 54.0 ↑ 13.0

Table 3: Performance comparison on the Linear Rule Learning benchmark using QWEN3-8B-IT.

#### 5.4 Analysis

Applicability to Exact-Memory Scenarios. MemFT is tailored for exact-memory tasks by addressing the token-level bottlenecks that govern verbatim reproduction. As shown in Table 4, many practical scenarios necessitate exact recall rather than semantic approximation, where a single token error can compromise validity or operational meaning. By reallocating gradient budget from mastered tokens to those below the deterministic recall threshold, MemFT enhances memory capacity, particularly under constrained LoRA capacity.

Beyond Memorization: Enhanced Generalization. To investigate whether MemFT’s focus on exact memorization compromises generalization, we introduce a Linear Rule Learning benchmark

Scenario Query Target Personal Credentials What is the login email and password

xxx.xxx@company.com/P@ssw0rd_xxx Legal Compliance Recite the exact wording of Article

for the internal portal?

Processing shall be lawful only if and to the extent that at least one of the following applies: the data subject has given consent...

5(1)(a) of the GDPR?

Medical Coding What is the ICD-10 code for uncomplicated Type 2 diabetes mellitus?

E11.9 Model Watermark Output the embedded ownership

MEM-2026-LoRA-EXACT-0x7F9A3B-COPYRIGHT Cloud Configuration What is the full endpoint of the pro-

watermark of this fine-tuned model?

s3:// prod-application-logs-123456789012-us-east-1

duction AWS S3 log bucket?

Academic Citation Output the complete LaTeX source code for cross-entropy loss?

\mathcal{L}_{CE}=-\sum_{i=1}^{C}y_i\ log(\hat{y}_i)

Security Secret What is the secret string used for memory leakage detection?

SECRET-TEST-XXXX-XXXX-XXXX-NONCE-1234 Software License What is the activation key for the

ENT-2026-ABCD-EFGH-IJKL-MNOP-QRST

enterprise edition license?

Table 4: Representative exact-memory scenarios across 8 domains. All tasks require verbatim recall because even minor deviations, such as a single-character, punctuation, or formatting error, can invalidate the target, alter its operational meaning, or introduce legal/security risks.

where the model learns f(x,y) = 3x+5y+7 with x,y ∈ [1,30]. The dataset comprises 500 training samples, with evaluation sets of 100 samples each for Exact Memory (seen pairs) and Generalization (unseen pairs). For both evaluation sets, we report the accuracy of correct answers. As shown in Table 3, MemFT consistently outperforms SFT in generalization accuracy, with gains ranging from 7%–15% across ranks. We attribute this gain to MemFT’s mitigation of overconfidence on easy samples and its prioritization of “stubborn tokens,” which enhances generalization.

### 6 Related Work

LLM Memory. LLM memory strategies are broadly categorized into non-parametric and parametric methods. Non-parametric methods, including In-Context Learning (ICL) (Brown et al., 2020), Retrieval-Augmented Generation (RAG) (Lewis et al., 2020), and sophisticated external memory systems (Packer et al., 2023; Liu et al., 2023; Zhong et al., 2024; Tan et al., 2025b; Fang et al., 2025; Chhikara et al., 2025; Kang et al., 2025), inject information at inference time. However, these approaches remain fundamentally constrained by fixed context windows and attention dilution (Liu et al., 2024; Kuratov et al., 2024; Bai et al., 2024). Even with long-context optimizations (Xiao et al., 2024; Xu et al., 2025; Li et al., 2026), they inherently decouple memory storage from parametric knowledge. In contrast, parametric memory incorporates knowledge directly into model parameters or modular parameter structures, enabling per-

sistent storage and retrieval-free reasoning (Meng et al., 2022; Yang et al., 2024; Li et al., 2025; Lei et al., 2026). However, existing studies primarily evaluate memory through downstream functional tasks (Maharana et al., 2024; Wu et al., 2025), leaving the quantitative patterns and mechanisms of parametric memory capacity largely unexplored.

LoRA as Parametric Memory. Low-Rank Adaptation (LoRA) (Hu et al., 2022) is widely used for parameter-efficient fine-tuning (Zhang et al., 2023; Ostapenko et al., 2024) and has recently been adopted as a modular memory mechanism for encoding new knowledge (Pletenev et al., 2025; Tan et al., 2025a; Chen et al., 2025; Charakorn et al., 2025; Liang et al., 2025; Charakorn et al., 2026; Back et al., 2026). Prior work mainly demonstrates its effectiveness through downstream task performance improvements (Jukic et al., 2025; Abdalla et al., 2025; Xu et al., 2026) and synergy with external memory systems (Su et al., 2025; Back et al., 2026). In contrast, we use LoRA as a controllable probe for studying parametric memory, focusing on its quantitative capacity laws and mechanisms.

### 7 Conclusion

By leveraging LoRA as a controllable probe into the memory mechanisms within the latent space of LLMs, we uncover the Parametric Memory Law. This law characterizes loss reduction as a powerlaw function of both LoRA rank and sequence length. We further reveal a deterministic phase transition in token-level loss dynamics, where un-

resolved bottleneck tokens can trigger decoding collapse. Guided by this mechanistic understanding, we propose MemFT, a fine-tuning strategy designed to explicitly resolve these critical memory bottlenecks.

### Limitations

While our work elucidates the parametric memory laws and phase transitions, several limitations persist. First, our analysis is restricted to 8B-scale models, leaving the generalization of the Parametric Memory Law to other scales unverified. Sec-

- ond, the p = 0.5 phase transition is specific to greedy decoding; its robustness under stochastic methods (e.g., nucleus sampling) remains to be verified. Finally, while we provide a preliminary generalization analysis, a comprehensive assessment of trade-offs with broader capabilities like open-ended reasoning is still lacking. Ethics Statement

We acknowledge the dual-use nature of memorization techniques, which could potentially be misused to encode harmful content. However, our study focuses exclusively on understanding the mechanistic boundaries of model capacity. All experiments rely on standard benchmarks that contain no sensitive personal information, and the examples in Table 4 are purely synthetic artifacts created for illustration. By clarifying these capacity limits, we aim to contribute to safer model design, though we emphasize that responsible deployment requires context-aware implementation and adherence to established safety guidelines.

### References

M. H. I. Abdalla, Zhipin Wang, Christian Frey, Steffen Eger, and Josif Grabocka. 2025. Zhyper: Factorized hypernetworks for conditioned LLM fine-tuning. CoRR, abs/2510.19733.

Seungju Back, Dongwoo Lee, Naun Kang, Taehee Lee, S. K. Hong, Youngjune Gwon, and Sungjin Ahn. 2026. Understanding lora as knowledge memory: An empirical analysis. CoRR, abs/2603.01097.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:

Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3119–3137. Association for Computational Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, and Robert Tjarko Lange. 2025. Text-to-lora: Instant transformer adaption. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, and Robert Tjarko Lange. 2026. Doc-to-lora: Learning to instantly internalize contexts. CoRR, abs/2602.15902.

Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Benjamin Van Durme, Luke Zettlemoyer, Jianfeng Gao, and Hao Cheng. 2025. Generative adapter: Contextualizing language models in parameters with A single forward pass. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. In ECAI 2025 - 28th European Conference on Artificial Intelligence, 25-30 October 2025, Bologna, Italy - Including 14th Conference on Prestigious Applications of Intelligent Systems (PAIS 2025), Frontiers in Artificial Intelligence and Applications, pages 2993–3000. IOS Press.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. 2024. Language modeling is compression. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A survey on in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in

Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 1107–1128. Association for Computational Linguistics.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2025. Lightmem: Lightweight and efficient memory-augmented generation. CoRR, abs/2510.18866.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. CoRR, abs/2312.10997.

Zihong He, Weizhe Lin, Hao Zheng, Fan Zhang, Matt W. Jones, Laurence Aitchison, Xuhai Xu, Miao Liu, Per Ola Kristensson, and Junxiao Shen. 2024. Human-inspired perspectives: A survey on AI longterm memory. CoRR, abs/2411.00489.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. 2024. Repeat after me: Transformers are better than state space models at copying. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, Proceedings of Machine Learning Research, pages 21502–21521. PMLR / OpenReview.net.

Josip Jukic, Martin Tutek, and Jan Snajder. 2025. Context parametrization with compositional adapters. CoRR, abs/2509.22158.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 25961–25970. Association for Computational Linguistics.

Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Y. Sorokin, and Mikhail Burtsev. 2024. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, and Soujanya Poria. 2026. δ-mem: Efficient online memory for large language models.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020.

Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Jiakai Li, Rongzheng Wang, Yizhuo Ma, Shuang Liang, Guangchun Luo, and Ke Qin. 2026. Dsas: A universal plug-and-play framework for attention optimization in multi-document question answering. Advances in Neural Information Processing Systems, 38:174538–174564.

Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Jihao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhiqiang Yin, Qingchen Yu, Bo Tang, Hongkang Yang, Zhi-Qin John Xu, and Feiyu Xiong. 2025. Memos: An operating system for memoryaugmented generation (MAG) in large language models. CoRR, abs/2505.22101.

Zhiyuan Liang, Dongwen Tang, Yuhao Zhou, Xuanlei Zhao, Mingjia Shi, Wangbo Zhao, Zekai Li, Peihao Wang, Konstantin Schürholt, Damian Borth, Michael M. Bronstein, Yang You, Zhangyang Wang, and Kai Wang. 2025. Drag-and-drop llms: Zero-shot prompt-to-weights. CoRR, abs/2506.16406.

Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang. 2023. Thinkin-memory: Recalling and post-thinking enable llms with long-term memory. CoRR, abs/2311.08719.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 13851–13870. Association for Computational Linguistics.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Oleksiy Ostapenko, Zhan Su, Edoardo M. Ponti, Laurent Charlin, Nicolas Le Roux, Lucas Caccia, and Alessandro Sordoni. 2024. Towards modular llms by building and reusing a library of loras. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024,

Proceedings of Machine Learning Research, pages 38885–38904. PMLR / OpenReview.net.

Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E. Gonzalez. 2023. Memgpt: Towards llms as operating systems. CoRR, abs/2310.08560.

Sergey Pletenev, Maria Marina, Daniil Moskovskiy, Vasily Konovalov, Pavel Braslavski, Alexander Panchenko, and Mikhail Salnikov. 2025. How much knowledge can you pack into a lora adapter without harming llm? In Findings of the Association for Computational Linguistics: NAACL 2025, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, Findings of ACL, pages 4309–4322. Association for Computational Linguistics.

V.F. Reyna and C.J. Brainerd. 1995. Fuzzy-trace theory: An interim synthesis. Learning and Individual Differences, 7(1):1–75. Special Issue: Fuzzy-Trace Theory.

Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented generation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2025, Padua, Italy, July 13-18, 2025, pages 1240–1250. ACM.

Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. 2025a. Better wit than wealth: Dynamic parametric retrieval augmented generation for test-time knowledge enhancement. CoRR, abs/2503.23895.

Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Rajan Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. 2025b. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 8416–8439. Association for Computational Linguistics.

Llama Team. 2024. The llama 3 herd of models. CoRR, abs/2407.21783.

Qwen Team. 2025. Qwen3 technical report. CoRR, abs/2505.09388.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In The Twelfth

International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-MEM: agentic memory for LLM agents. CoRR, abs/2502.12110.

Ziwen Xu, Chenyan Wu, Hengyu Sun, Haiwen Hong, Mengru Wang, Yunzhi Yao, Longtao Huang, Hui Xue, Shumin Deng, Zhixuan Chu, Huajun Chen, and Ningyu Zhang. 2026. Why steering works: Toward a unified view of language model parameter dynamics. CoRR, abs/2602.02343.

Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu, Zhiyu Li, Bo Tang, Wenqiang Wei, Jinbo Wang, Zeyun Tang, Shichao Song, Chenyang Xi, Yu Yu, Kai Chen, Feiyu Xiong, Linpeng Tang, and Weinan E. 2024. Memory3: Language modeling with explicit memory. CoRR, abs/2407.01178.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems, methods, and opportunities. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 10222–10240. Association for Computational Linguistics.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2).

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 19724– 19731. AAAI Press.

Tongyao Zhu, Qian Liu, Liang Pang, Zhengbao Jiang, Min-Yen Kan, and Min Lin. 2024. Beyond memorization: The challenge of random memory access in language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3373– 3388. Association for Computational Linguistics.

### A Dataset Construction Details

We use two controlled benchmarks to evaluate exact parametric memory: the Long-Context Memorization Stress Test and the PhoneBook benchmark.

The Long-Context Memorization Stress Test is designed to probe memory capacity under longsequence settings, consisting of synthetic key-value pairs with no overlap with pretraining data. Specifically, we randomly sample a sequence from the LongBench dataset and encode it using the Qwen tokenizer. We then randomly replace 0%, 20%, 40%, 60%, 80%, and 100% of the tokens with random tokens from the Qwen vocabulary to serve as value contents with varying semantic coherence and difficulty levels. Each instance is paired with the fixed key: “Please output the content of the vector memory injected into the activations.”

For the PhoneBook benchmark, we adopt and adapt the standard version for parametric memorization evaluation. The original dataset is structured with three fields: context, query(key), and target(value), where context contains namephone mappings, query specifies the queried name, and target provides the corresponding phone number. Since our work studies parametric memorization rather than in-context retrieval, we remove the context field entirely and retain only the keyvalue pairs. During preprocessing, we deduplicate all key entries to eliminate conflicting key-value associations, ensuring each query maps to exactly one unique value.

We construct length buckets using answer-only token counts. For each processed key-value pair, we tokenize only the value string with the model-specific tokenizer and exclude prompt, chattemplate, and key tokens from the length budget. We accumulate pairs until the total number of answer tokens reaches the desired bucket size L. Due to the highly regular structure of phone number values, bucket boundaries can be matched exactly in all experiments.

### B LoRA Configuration and Rank Settings

All experiments freeze the full base model parameters and train only the LoRA adapters. The LoRA rank is treated as the primary controllable parameter budget for studying exact parametric memory.

For the Long-Context Memorization Stress Test experiments, LoRA is applied only to the MLP down_proj module. For PhoneBook experiments, LoRA is applied once to the entire MLP block.

For layer selection, the Long-Context Memorization Stress Test uses layers 20 and 24 for Qwen38B-Instruct, and layers 18 and 20 for Llama3.18B-Instruct. For PhoneBook, we use layer 24 for Qwen3-8B-Instruct and layer 18 for Llama3.1-8BInstruct.

### C Aggregation Protocol for Main Results

The main result tables report rank-wise performance. For each reported rank, we evaluate the corresponding LoRA adapter across a set of length buckets and report the average accuracy over these buckets. Therefore, each rank column reflects the average memorization performance under a fixed LoRA rank across multiple memory lengths, rather than the result from a single length.

For the Long-Context Memorization Stress Test, each reported rank-wise accuracy is averaged over the following length buckets:

LLong = {50,100,200,500,1000,2000,3000, 4000,5000,6000,7000,8000,10000}.

For PhoneBook, each reported rank-wise accuracy is averaged over the following answer-only length buckets:

LPB = {1k,2k,4k,8k,12k,16k,24k,32k}. These buckets follow the PhoneBook answer-only length accounting described in Appendix A.

### D PhoneBook Inter-Batch Curriculum Hyperparameters

For PhoneBook experiments using MemFT-SW, we apply the Inter-Batch Temporal Curriculum with length-dependent hyperparameters. All curriculum schedules use the same exposure ratios:

##### [0.2,0.4,0.6,0.8,1.0].

The boundary list specifies the epoch at which the training curriculum moves to the next exposure ratio. For example, the boundary list [20,40,60,80,300] means that the model uses 20%, 40%, 60%, 80%, and 100% of the PhoneBook training pairs over the corresponding epoch intervals.

Because PhoneBook targets are tokenized differently by Qwen3-8B-Instruct and Llama3.1-8BInstruct, the same answer-only length may correspond to different numbers of training pairs. We therefore report the length-dependent curriculum hyperparameters separately for the two models in Tables 5– 6

###### Length Approx. Samples LR Epochs Batch Size Grad. Accum. Curriculum Boundaries

- 1k 100 1e−2 300 10 1 [20, 40, 60, 80, 300]
- 2k 200 1e−2 300 10 1 [20, 40, 60, 80, 300] 4k 400 1e−2 350 10 1 [20, 40, 60, 80, 350] 8k 800 7e−3 350 20 2 [30, 60, 90, 120, 350]

12k 1200 5e−3 500 40 2 [60, 120, 180, 240, 500] 16k 1600 5e−3 600 40 2 [80, 160, 240, 320, 600] 24k 2400 5e−3 600 40 2 [80, 160, 240, 320, 600] 32k 3200 5e−3 700 40 2 [100, 200, 300, 400, 700]

- Table 5: Length-dependent curriculum hyperparameters for Qwen3-8B-Instruct on the PhoneBook benchmark using MemFT-SW. The approximate sample count is computed from the answer-only PhoneBook tokenization under the Qwen tokenizer. All schedules use curriculum exposure ratios [0.2,0.4,0.6,0.8,1.0].

Length Approx. Samples LR Epochs Batch Size Grad. Accum. Curriculum Boundaries

- 1k 250 1e−2 300 25 1 [20, 40, 60, 80, 300]
- 2k 500 1e−2 400 25 1 [40, 80, 120, 160, 400] 4k 1000 7e−3 400 50 1 [40, 80, 120, 160, 400] 8k 2000 5e−3 600 50 2 [80, 160, 240, 320, 600]

12k 3000 5e−3 700 50 2 [100, 200, 300, 400, 700] 16k 4000 5e−3 700 50 2 [100, 200, 300, 400, 700] 24k 6000 3e−3 700 50 4 [100, 200, 300, 400, 700] 32k 8000 3e−3 800 50 4 [120, 240, 360, 480, 800]

- Table 6: Length-dependent curriculum hyperparameters for Llama3.1-8B-Instruct on the PhoneBook benchmark using MemFT-SW. The approximate sample count is computed from the answer-only PhoneBook tokenization under the Llama tokenizer. Since PhoneBook targets are shorter under the Llama tokenizer, the same answeronly length corresponds to more training pairs than in Qwen. All schedules use curriculum exposure ratios [0.2,0.4,0.6,0.8,1.0].

### E Additional Training Convergence Results

We provide additional training-loss curves to verify that the reported LoRA adapters are trained to convergence across the full rank–length sweep. Each subfigure corresponds to one fixed length– rank configuration, and curves show the training loss trajectory under the corresponding dataset/model setting. These plots are intended to rule out under-training as an explanation for the observed differences in downstream memory accuracy.

### F Additional Performance Landscapes

We further visualize the full performance landscape across LoRA ranks, memory lengths, models, and training methods. For each model, subplots are grouped by LoRA rank; within each subplot, curves compare different training methods as memory length increases. These figures complement the averaged results in the main tables by showing where each method gains or loses performance

across the rank–length grid.

### G Token-Level Probability Grids Across Data Scenarios

We provide full per-position teacher-forcing probability grids p(ti | t<i) for Qwen3-8B (layer 24) across three representative data scenarios of the Long-Context Memorization Stress Test (Figures 10–13). Each grid has rows indexed by memory length L and columns indexed by LoRA rank r. Blue curves show the token-level probability; red dots mark positions where p < 0.5 (stubborn positions); black dotted vertical lines indicate the free-run first-failure position i∗.

r = 4

r = 8

r = 16

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10

L=50

5

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10

L=100

5

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10

L=200

5

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10

L=500

5

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=1000

10

5

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=2000

10

5

TrainingLoss

0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=3000

10

5

0

| | | | | |
|---|---|---|---|---|
| | | | | |
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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=4000

10

5

0

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=5000

10

5

0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=6000

10

5

0

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

L=7000

10

5

0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

L=8000

10

5

0

L=10000

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10

5

0

0 1000 2000

0 1000 2000 0 1000 2000 0 1000 2000

0 1000 2000

Global Step

- Figure 4: Training convergence of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Each subplot corresponds to a fixed memory length and LoRA rank. The overlaid curves represent different random-token mixture settings. The consistent decrease and stabilization of training loss indicate that the LoRA adapters are sufficiently optimized across the full sweep.

r = 4 r = 8 r = 16

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

7.5

L=50

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

7.5

L=100

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

7.5

L=200

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

7.5

L=500

5.0

2.5

0.0

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

L=1000

7.5

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

L=2000

7.5

5.0

2.5

TrainingLoss

0.0

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

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

L=3000

7.5

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

L=4000

7.5

5.0

2.5

0.0

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

L=5000

7.5

5.0

2.5

0.0

| | | | | |
|---|---|---|---|---|
| | | | | |
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

L=6000

7.5

5.0

2.5

0.0

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

L=7000

7.5

5.0

2.5

0.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
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

L=8000

7.5

5.0

2.5

0.0

L=10000

| | | | | |
|---|---|---|---|---|
| | | | | |
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

7.5

5.0

2.5

0.0

0 1000 2000

0 1000 2000

0 1000 2000

0 1000 2000

0 1000 2000

Global Step

- Figure 5: Training convergence of Llama3.1-8B on the Random / Long-Context Memorization Stress Test. The figure follows the same layout as Figure 4, with each subplot corresponding to a fixed length–rank configuration. The curves show stable optimization behavior across random-token mixture settings, supporting the reliability of the subsequent accuracy comparisons.

| | |
|---|---|
| | |

SFT MemFT-OT

r = 1

r = 2 r = 4

r = 8 r = 16 r = 32 r = 64

4

L=1000

2

0

4

L=2000

2

0

4

L=4000

2

0

4

L=8000

TrainingLoss

2

0

L=12000

4

2

0

L=16000

4

2

0

L=24000

4

2

0

L=32000

4

2

0

0 50000

0 50000 0 50000 0 50000 0 50000 0 50000 0 50000

Global Step

- Figure 6: Training convergence of Llama3.1-8B on the PhoneBook benchmark. Each subplot corresponds to a fixed answer-token length and LoRA rank. The curves compare SFT and MemFT-OT under the same configuration, showing that the PhoneBook runs are sufficiently optimized before evaluating exact-match recall.

| | | | | | | |
|---|---|---|---|---|---|---|
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
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 6

r = 8

r = 10

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 14

r = 16

r = 24

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 32

r = 48

r = 64

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 128

r = 256

r = 512

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

50 100 200 5001,0002,000 5,000

50 100 200 5001,0002,000 5,000

50 100 200 5001,0002,000 5,000

Memory Length (Tokens)

Memory Length (Tokens)

Memory Length (Tokens)

Method

MemFT-OT MemFT-SW SFT

- Figure 7: Exact-match accuracy of Qwen3-8B on the Random / Long-Context Memorization Stress Test. Each subplot corresponds to one LoRA rank, the x-axis denotes memory length, and the y-axis denotes exact-match accuracy. Curves compare SFT, MemFT-OT, and MemFT-SW, showing how each method scales with increasing memory length under a fixed rank.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 6

r = 8

r = 10

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 14

r = 16

r = 24

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 32

r = 48

r = 64

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

r = 128

r = 256

r = 512

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.00

0.75

Acc

0.50

0.25

0.00

50 100 200 5001,0002,000 5,000

50 100 200 5001,0002,000 5,000

50 100 200 5001,0002,000 5,000

Memory Length (Tokens)

Memory Length (Tokens)

Memory Length (Tokens)

Method

MemFT-OT MemFT-SW SFT

- Figure 8: Exact-match accuracy of Llama3.1-8B on the Random / Long-Context Memorization Stress Test. The layout matches Figure 7: each subplot fixes one LoRA rank and compares SFT, MemFT-OT, and MemFT-SW across memory lengths.

r = 1

r = 2

r = 4

r = 8

1.00

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.75

Acc Qwen3-8B-IT

0.50

0.25

0.00

r = 16

r = 32

r = 64

1.00

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.75

Acc

0.50

0.25

0.00

r = 1

r = 2

r = 4

r = 8

1.00

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.75

Acc Llama3.1-8B-IT

0.50

0.25

0.00

r = 16

r = 32

r = 64

1.00

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
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.75

Acc

0.50

0.25

0.00

1,0002,0004,0008,00016,00032,000

1,0002,0004,0008,00016,00032,000

1,0002,0004,0008,00016,00032,000

Memory Length (Tokens)

Memory Length (Tokens)

Memory Length (Tokens)

Method

MemFT-OT MemFT-SW SFT

- Figure 9: Exact-match accuracy on the PhoneBook benchmark for Qwen3-8B and Llama3.1-8B. The upper panels correspond to Qwen3-8B and the lower panels correspond to Llama3.1-8B. Each subplot fixes one LoRA rank and plots exact-match accuracy as a function of answer-token length. Curves compare SFT, MemFT-OT, and MemFT-SW, providing a full view of how model, rank, length, and training method jointly affect short key–value memorization.

[Figure 7]

###### Figure 10: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 100% scenario with r ∈ {8,10,12,14,16}. This rank range aligns with the LongBench-mixed scenarios for direct comparison.

[Figure 8]

###### Figure 11: Per-position probability grid for the Random (100%) scenario with r ∈ {48,64,128,256,512}.

[Figure 9]

###### Figure 12: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 20% scenario. With 80% semantically coherent tokens from LongBench, the model memorizes more easily and stubborn positions appear only at longer lengths.

[Figure 10]

###### Figure 13: Per-position probability grid for the Long-Context Memoriza tion Stress Test Random 60% scenario. With 40% semantically coherent tokens, the difficulty is intermediate between the Random 100% and Random 20% settings, and stubborn positions emerge at shorter lengths compared to Figure 12.

