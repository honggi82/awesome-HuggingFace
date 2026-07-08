arXiv:2506.19830v1[cs.LG]24Jun2025

# Scaling Speculative Decoding with LOOKAHEAD REASONING

Yichao Fu1 Rui Ge2 Zelei Shao3 Zhijie Deng2 Hao Zhang1 1UCSD 2Shanghai Jiao Tong University 3UIUC

## Abstract

Reasoning models excel by generating long chain-of-thoughts, but decoding the resulting thousands of tokens is slow. Token-level speculative decoding (SD) helps, but its benefit is capped, because the chance that an entire γ-token guess is correct falls exponentially as γ grows. This means allocating more compute for longer token drafts faces an algorithmic ceiling – making the speedup modest and hardware-agnostic. We raise this ceiling with LOOKAHEAD REASONING, which exploits a second, step-level layer of parallelism. Our key insight is that reasoning models generate step-by-step, and each step needs only to be semantically correct, not exact token matching. In LOOKAHEAD REASONING, a lightweight draft model proposes several future steps; the target model expands each proposal in one batched pass, and a verifier keeps semantically correct steps while letting the target regenerate any that fail. Token-level SD still operates within each reasoning step, so the two layers of parallelism multiply. We show LOOKAHEAD REASONING lifts the peak speedup of SD both theoretically and empirically. Across GSM8K, AIME, and other benchmarks, LOOKAHEAD REASONING improves the speedup of SD from 1.4x to 2.1x while preserving answer quality, and its speedup scales better with additional GPU throughput. Our code is available at https://github.

com/hao-ai-lab/LookaheadReasoning

## 1 Introduction

Large reasoning models (LRMs) have recently pushed the state of the art in math problem solving and program synthesis by generating explicit, long Chains of Thoughts (CoT) [1]. In these models, an answer unfolds as a sequence of intermediate reasoning “steps”, and each step arrives token by token via autoregressive decoding. If a solution needs N steps and each step needs T tokens, the model must generate O(NT) tokens, often running into tens of thousands of tokens and minutes of wall-clock time. For instance, OpenAI’s o1 model [2] may take more than 2 minutes to solve a single problem from the International Mathematical Olympiad (IMO) challenges.

Speculative decoding (SD) mitigates this token-level dependency by spending additional FLOPs to shorten the critical path of generation: a cheap draft model proposes γ future tokens and the expensive target model then verifies them in parallel; if every guess matches, the decoding can fast forward γ + 1 positions at once. However, in the face of LRMs with long decode, two facts limit how far this idea can scale. First, the probability of an entire γ-token sequence is correct drops almost exponentially with γ (§2), so the expected number of accepted tokens quickly saturates as γ grows. Second, the verifier must still verify the target logits for all γ positions, and that cost grows linearly. This results in a speedup curve that climbs with small γ, plateaus after a few dozen tokens, and can even decline once the verification cost dominates. For example, in a real profiling, we observe SD’s speedup caps at 1.4x (Figure 2). As this ceiling is algorithmic rather than hardware-bound, this means that allocating more FLOPs in SD yields only diminishing returns, making SD’s acceleration not scale with future accelerators. As LRMs produce ever-longer generations, the number of tokens SD can safely skip does not grow proportionally, so the end-to-end latency remains substantial.

Preprint. Under review.

[Figure 1]

- Figure 1: One cycle of LOOKAHEAD REASONING. The draft model proposes γ = 3 steps {sˆ1, sˆ2, sˆ3}. The target model then generate {s1, s2, s3} based on prefixes and {sˆ1, sˆ2, sˆ3}, respectively. Verifier checks if draft and target steps are semantically equivalent (e.g., s1 ≈ sˆ1). If the first two steps are equivalent but the third is not, LOOKAHEAD REASONING outputs the verified draft steps

(sˆ1, sˆ2) followed by the target’s correction (s3). This allows accepting multiple steps with only a lowered latency (e.g., 2t + T) compared to the sequential target calls in autoregressive decoding (e.g., 3T), where t is draft step time and T is target step time.

0 50 100 150 200

# draft tokens

1.2

1.4

1.6

1.8

2.0

Speedup

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

Ours + NGRAM

NGRAM

- Figure 2: Speedup vs Draft Tokens. Speedup over autoregressive decoding, comparing LOOKAHEAD REASONING combined with token-level SD (NGram-based) (red line) to SD alone (blue line). Our method is orthogonal to token-level SD and improves the maximum speedup from 1.4× to 2.1×.

This paper makes a key observation that reasoning is naturally hierarchical: a full chain-of-thought breaks into discrete steps, and each step unrolls tokens by token. To reach the correct answer, a reasoning step requires only semantic correctness but not exact token matches. To illustrate, we replaced over 50% of DeepSeek-R1 32B’s reasoning steps with semantically equivalent ones from another smaller model. The impact on overall task accuracy was minimal, with deviations typically not exceeding 2% (§4.1). This looser requirement exposes a coarser unit for speculation: in addition to guessing the next few tokens, a model can guess and verify the next few reasoning steps. These step proposals and verification are independent, so they can be batched and executed in parallel, making full use of GPU’s batching capacity. At the same time, token-level speculation can still operate within each step, achieving two complementary layers of parallelism rather than one.

This paper develops LOOKAHEAD REASONING based on this insight, with one operational cycle shown in Figure 1. First, a lightweight draft model autoregressively generates several sequential, future reasoning steps {sˆ1,sˆ2,sˆ3}. Concurrently, the target LRM generates corresponding followup steps {s1,s2,s3}, where each si is generated based on a prefix formed by the initial context concatenated with the sequence of preceding draft steps sˆ1,...,sˆi−1. Notably, the generations of {s1,s2,s3} are issued as a batch running in parallel to exploit additional cross-request parallelism on GPUs. A lightweight verifier, implemented as a small LLM-as-a-Judge or an embedding model, then begins with the first speculative step to determine if the draft’s original speculative step sˆ1 semantically aligns with this orcale step s1. If the step passes verification, we keep it and proceed to the verification of sˆ2 and s2, which are already available due to batched execution. If it fails, we drop it and revert to the target model’s standard generation. Concurrently, token-level speculation can operate independently when the target/draft model generates the content of each step.

LOOKAHEAD REASONING operates at step level, an axis orthogonal to token-level speculation. Because step-level speculation absorbs compute that would otherwise hit the speedup ceiling by token-level speculation, the method scales better with hardware. Additional FLOPs can draft more (or deeper) steps instead of lengthening token guesses, sidestepping diminishing returns faced by token-level-only speculative decoding. For example, on GSM8K, LOOKAHEAD REASONING lifts token-level SD’s peak speedup from 1.4x to 2.1x (combined), as depicted in Figure 2. Even when available compute is limited, we prove that the peak speedup given limited compute shall be achieved only by combining both levels of speculation (§3.3).

A key design consideration is the choice of verifier. While an ideal semantic verifier ensures no accuracy loss, practical ones balance compute cost against judgment accuracy. For instance, a looser verification may boost draft acceptance (and speedup) but risks accuracy drop from erroneous steps. We finally opted for a 7B LLM-As-a-Judge, striking a balance between these competing factors (§4.3).

To sum up, our contributions can be listed as follows: (1) We develop LOOKAHEAD REASONING, a novel step-level speculation dimension to scale speculative decoding, orthogonal to existing tokenlevel approaches. (2) We present theoretical analysis demonstrating significant speedups from our method, both as a standalone technique and when combined with token-level SD. (3) We conduct extensive experiments showing consistent performance improvements across diverse datasets.

## 2 Background

Speculative Decoding. LLMs autoregressively generate one token at a time, with each next token xt+1 sampled from the distribution P(xt+1 | x1:t). This sequential dependency poses a fundamental bottleneck to inference speed. Speculative decoding [3] mitigates this challenge using a “guess-and-verify” strategy with two models: a lightweight draft model q and a strong target model p. Given a context x1:t, q autoregressively proposes a sequence of γ candidates tokens, xˆt+1:t+γ, along with their draft probabilities Q. Subsequently, p verifies these γ tokens in a single parallel forward pass, yielding the target probabilities P. A rejection-sampling procedure then sequentially processes each proposed token xˆt+i. If a token is accepted, it is appended to the output; if rejected, the process halts, and a final token is sampled from p’s distribution based on the last accepted token. This allows for the acceptance of n ≤ γ tokens in fewer steps than standard autoregression.

The theoretical speedup achieved by this speculative decoding approach, assuming negligible verification overhead beyond the target model’s single pass, can be characterized by:

1 − αγ+1 (1 − α)(1 + cγ)

g(γ) =

,

where α represents the average acceptance rate of each token drafted by q (assuming it’s independent for each token in the sequence). Note that the probability of an entire sequence of γ tokens being accepted typically decreases exponentially with γ, as errors accumulate. c = Tq/Tp is the latency ratio of generating a single token by the draft model relative to the target model, where Tq and Tp are step time of q and p, respectively. Furthermore, for all c ≥ 0 and γ ≥ 0, the speedup g(γ) is upper-bounded by 1/(1 − α), a limit approached only in the idealized case where c = 0. This inherent upper bound on g(γ) signifies a critical limitation: beyond an optimal point, investing more computational resources by increasing the speculative length (γ) yields diminishing or even negative returns on speedup. Therefore, this algorithmic ceiling implies that the acceleration gains from token-level speculative decoding do not scale with improvements in hardware, such as more powerful GPUs. Consequently, for reasoning models with longer CoTs, the bounded acceleration offered by token-level speculative decoding alone highlights an urgent need for more potent acceleration strategies.

LLM Reasoning. Large reasoning models (LRMs) [2, 4] are increasingly pivotal for complex tasks such as math problem-solving and coding. These models often generate solutions by giving a "chainof-thought" (CoT)—a sequence of intermediate reasoning steps, denoted as s, produced step-by-step to derive a final answer [1]. Each reasoning step (si) typically conditions on the previous one (si−1), creating a sequential dependency analogous to token-level autoregression but at a higher conceptual granularity. We observe that this step-wise structure itself presents a significant opportunity for acceleration. Specifically, entire reasoning steps can be speculatively proposed by a draft model,

denoted as sˆ. Our preliminary experiments (§4.1) show this potential. A small 1.5B draft model can generate speculative steps sˆ that semantically align with over 50% of ground-truth steps s from a much larger 32B target model. Besides, this is achieved while maintaining comparable task accuracy.

## 3 Method

In this section, we explain the details of LOOKAHEAD REASONING, then provide theoretical analysis that shows its performance benefits. Furthermore, since both step-level and token-level speculative generation rely on increasing concurrency, we show that in real-world settings—where the two methods compete for limited concurrency resources—peak performance gains can only be achieved when combining both speculative strategies together.

### 3.1 LOOKAHEAD REASONING: Step-Level Speculative Generation Solution

The core idea of LOOKAHEAD REASONING is to perform speculation and verification on entire steps rather than individual tokens. To put it clear, we first presented a synchronous version of this approach in Algorithm 1, and then conceptually illustrate an optimized asynchronous variant in Figure 1.

As detailed in Algorithm 1 (sync version), one cycle of LOOKAHEAD REASONING proceeds as follows:

- 1. Draft Step Generation: Given token prefix x1:t, the draft model q first autoregressive generates a sequence of γ candidate steps, denoted as sˆ0,...,sγˆ−1. Each step sˆj is generated conditioned on

the prefix extended by all preceding draft steps: x1:t ⊕ k j−=01 sˆk. In practice, we simply use ‘\n\n’ as the step break as we found it’s a common flag for step split in various reasoning models [5, 4, 6].

- 2. Parallel Target Step Generation: Same as in the standard speculative decoding [3], once all γ draft steps are available, target model p generates following steps s0,...,sγ accordingly in parallel.
- 3. Verification and Output Construction: The algorithm then determines the longest prefix of accepted draft steps. Verification between each draft step sˆj and its corresponding target step sj is performed by a verifier V (sj,sˆj), which assesses whether if sˆj is an acceptable substitute for sj, i.e. whether they are semantic similar. Algorithm 1 Lookahead Reasoning(Sync Version) Input: Draft model q, Target model p, Prefix x1:t, Max lookahead steps γ, Verifier V (·, ·) → {True, False}

- 1: Initialize empty step sequences sˆ0, . . . , sγˆ−1, s0, . . . , sγ
- 2: xcurrent =. x1:t
- 3: for j = 0 to γ − 1 do ▷ Generate γ draft steps sequentially
- 4: sˆj =. q.GenerateStep(xcurrent); xcurrent =. xcurrent ⊕ sˆj
- 5: in parallel do for j = 0 to γ: ▷ Compute target steps sj in parallel based on draft prefixes
- 6: Let x′j =. x1:t ⊕ jk−=01 sˆk if j ≥ 1 else x1:t ▷ Prefix before draft step j
- 7: sj =. p.GenerateStep(x′j)
- 8: end parallel
- 9: j∗ =. min({j ∈ {0..γ − 1} | V (sj, sˆj) == False} ∪ {γ}) ▷ Find first unaccepted step using V
- 10: OutputSequence =. ( j

∗−1

k=0 sˆk) ⊕ sj∗ ▷ Append verified drafts + decisive target Output: OutputSequence

Verifier Selection. The choice of verifier (V ) is a pivotal design consideration in LOOKAHEAD REASONING. While an ideal semantic verifier ensures no accuracy loss, practical implementations face a primary trade-off between judgment precision and computational overhead; Furthermore, the strictness of verification (e.g., a threshold) presents a secondary trade-off, potentially boosting draft acceptance and speedup at the risk of degrading task accuracy from erroneously accepted steps. We explore three common paradigms for semantic assessment—LLM-as-a-Judge [7] for nuanced evaluation, embedding-based verifier [8] for efficient similarity, and target model scoring [9]—each with distinct cost-precision profiles, empirically evaluating their impact in §4.3.

Asynchronous Generation (Illustrated in Figure 1. In Algorithm 1, the parallel verification steps launch only after all γ draft steps sˆj with j ∈ {0...γ − 1} are produced. An optimized asynchronous implementation, conceptually depicted in Figure 1, can begin generating a target step sj as soon as its required prefix (containing x1:t,sˆ0,...,sjˆ−1) becomes available from the draft model. This async execution brings overlap for draft/target generation and verification phases, which can significantly reduce end-to-end latency compared with the synchronous version. Note that in this asynchronous

mode implementation, both the draft and target models will “lookahead” γ steps, ensuring maximal utilization of parallel processing. This is different from the sync version that draft model generate γ drafts while target model generate γ + 1 steps in each cycle.

Multi-Branch Drafting. To further increase the number of the accepted reasoning steps, we explore tree-structure generation where the draft model proposes multiple candidate steps at each speculative position. Specifically, instead of generating a single candidate chain, the draft q can propose a set of W alternative steps for each position j in the draft sequence. Once a step is generated, the draft then proposes W child candidates in parallel for the subsequent position j + 1. This branching process continues up to a maximum γ steps, leading to an exponential growth in the total number of candidate sequences explores, i.e., Wγ. The target model p, however, still generate one single candidate continuation step for each position j (based on the draft prefix). The verifier V would then check if any of the W proposed draft branches for that position j semantically aligns with the target model’s step. If such a match is found, that branch is accepted and other branches are discarded. This multi-branch strategy aims to boost the likelihood of speculative success, albeit at the cost of increased computational effort in the drafting phase. We discussed its trade-off in §4.3.

### 3.2 Theoretical Speedup Analysis for LOOKAHEAD REASONING

We now analyze the potential performance gains of LOOKAHEAD REASONING. We make simplifying assumptions for clarity: negligible verification overhead, constant cost for generating steps, and a single draft branch at each stage.

Notation: Let γ1 be the maximum number of draft steps generated sequentially, and k1 = γ1 be the number of target steps that generate in parallel. Let T be the wall-time cost for the target model p to

generate one step, and c1T be the cost for the draft model q (0 < c1 < 1). Let α1 ∈ (0,1) be the probability that a draft step is accepted.

### Step-Level Speedup for Sync Version of LOOKAHEAD REASONING: The latency speedup for sync Lookahead Reasoning is

1 − αk

1

1

fsync(k1) =

.

(1 − α1)(1 − c1 + c1 k1)

Step-Level Speedup for Async Version of LOOKAHEAD REASONING: The speedup depends on whether the draft generation is limited by the maximum depth k1 or by the relative cost c1. Let Xi be the number of consecutively accepted draft steps in stage i. The expected number of accepted steps before a rejection is E[Xi] = α1/(1 − α1).

We define the asymptotic speedup S as the ratio of steps generated by LOOKAHEAD REASONING compared to a target-only baseline over the same wall-time. Two cases arise:

- 1. k1 ≥ ⌈1/c1⌉: The draft model is relatively slow, and generating k1 drafts takes longer than one target step. The depth limit k1 is effectively inactive. The speedup converges to:

S1 =

1 + E[Xi] 1 + c1E[Xi]

=

1 c1 + (1 − c1)(1 − α1)

- 2. k1 < ⌈1/c1⌉: The draft model is fast enough to generate k1 steps within time T. The maximum depth k1 limits the number of speculative steps per cycle. The speedup converges to:

S2 =

E[1 + Xi] E[⌈(Xi + 1)/k1⌉ + c1(Xi mod k1)]

=

1 − αk

1

1

(1 − α1) + c1 α1 − α1k+1 − k1(1 − α1)αk

1

1

(Detailed derivations are provided in Appendix B). Let fasync(k1) denote the step-level speedup, where fasync(k1) = S1 or S2 depending on the case.

- 3.3 Optimal Speculation Strategies under Concurrency Constraints

Token-level speculative decoding [3] and step-level speculative decoding are orthogonal to each other. If k2 is the number of additional tokens speculated by the draft model within each step generation

(for both draft and target models, assuming they use internal speculation) and c2 is the ratio of execution time of the draft model and target model in speculative decoding, its speedup is given by

g(k2), based on the token acceptance rate α2:

1 − αk

2

2

g(k2) =

(1 − α2)(1 − c + ck2)

Since these mechanisms operate at different granularities (inter-step vs. intra-step), their speedups multiply, yielding a combined speedup h(k1,k2) = f(k1) × g(k2). However, these two orthogonal parallelism dimensions compete for computational resources, making it crucial to determine the optimal resource allocation strategy to achieve maximum speedup. In this work, we focus on a fundamental question: is using both speculation methods superior to applying only one method?

### Optimality of Hybrid Approach under Budget Constraint:

In real-world systems, memory and computational constraints necessitate capping the total degree of parallelism (M) available to the target model, i.e., ParallelDimg × ParallelDimf for step-level and token-level speculative decoding methods, respectively. This constraint transforms our earlier question into a resource allocation optimization problem: given a finite parallel budget (M), should we distribute resources across both parallelism dimensions or concentrate them on a single method? Consequently, our design goal becomes:

h(k1,k2) = f(k1) × g(k2). (1) Where ParallelDimg = k2, and

max

ParallelDimg×ParallelDimf≤M

k1, sync min{⌈1c⌉,k1}, async

ParallelDimf =

It’s easy to see that if we set k1 = 1, then we are using purely token-level speculative decoding, whereas if we set k2 = 1, then we are using purely lookahead reasoning.

Theorem (Hybrid Method Optimality for Async Algorithm LOOKAHEAD REASONING): Under the conditions of acceptance rates (0.52 < α1,α2 < 0.8), reasonably efficient draft models (c1 < 1 3,c2 < 15), and sufficient parallelism budget M ≥ 16, the maximum speedup h(k1,k2) is achieved if and only if a hybrid strategy is employed, meaning both k1 ≥ 2 and k2 ≥ 2.

These conditions are broadly representative of real-world scenarios: acceptance rates (α1,α2) in the 0.52-0.8 range are common in speculative decoding [10] and our experiments (§4.1); draft

model efficiency ratios (c1) below 13 and (c2) below 15 are also common; and the parallelism budget (M ≥ 16) reflects typical GPU capabilities. This analysis demonstrates that neither pure step-level

nor pure token-level speculation is optimal under a fixed parallelism budget. The highest theoretical speedup is obtained by judiciously combining both strategies, leveraging parallelism across steps (k1) and within steps (k2). This motivates architectures and systems that can effectively manage and exploit both levels of speculative execution. It is empirically validated in §4.2. We provide a complete proof in Appendix B.1.2. Additionally, we analyze in detail the conditions under which single methods (either token-level or step-level) outperform hybrid approaches, and conversely, when combining both methods yields superior performance (Appendix B.1.2).

## 4 Experiment

Models. We evaluate two popular open-source reasoning model series: DeepSeek-R1-Distill [4] and Qwen3 [6]. For the DeepSeek-R1-Distill series, the 1.5B version serves as the draft model and the 32B version as the target model. Similarly, for the Qwen3 series, the 1.7B model is used as the draft model and the 32B model as the target. Unless otherwise specified, Qwen2.5-7B-Instruct [11] is employed as the judgement model. A deliberately designed judge prompt template allows our model to assess the semantic alignment between two sentences in just one prefill pass (Appendix A).

Datasets. Our evaluation spans a suite of benchmarks, aligning with previous speculative decoding research [12, 13] and reasoning model evaluations [4, 6]. For code generation, we use HumanEval [14] and LiveCodeBench [15]. Math reasoning tasks are assessed using GSM8K [16], AIME’24 [17], and

AMC12’23 [18]. For question answering, we include GPQA [19] and MT-Bench [7]. Specific to dataset sampling, we utilize 40 out of 50 problems from AMC12’23, selected by Qwen2.5 Math [20], and randomly sample 100 queries from the 1.3K GSM8K test set. For LiveCodeBench, We select 268 problems collected between August 2024 and Janaury 2025, following previous research [4].

General Parameters. LLM generation settings are configured specifically for each model series. For the DeepSeek-R1-Distill series, we adhere to the official settings with a temperature of 0.6, top_p of 0.95, and a maximum generation length of 32K. For the Qwen3 series, the temperature is set to 0.6, top_p to 0.95, min_p to 0, top_k to 20, and the maximum generation length is 37K. These maximum generation lengths are chosen to ensure complete outputs. We use prompt-lookup decoding (n-gram) [21] as a representative speculative decoding (SD) method: the max lookup tokens is set to 1 for GSM8K and 2 for other datasets. The number of speculative tokens is set to 8 for SD and the number of speculative steps is set to 6 for LOOKAHEAD REASONING by default.

Testbed. Experiments are conducted on a server equipped with eight NVIDIA H100 GPUs. Target models (32B) are deployed across two H100 GPUs using tensor parallelism. Draft models (1.5B/1.7B) and the default judge model (Qwen2.5-7B-Instruct) are each deployed on a single H100 GPU. Our algorithm is built upon the vLLM v0.8.3. Both the baseline and our proposed method are evaluated using vLLM [22] to simulate real-world LLM serving conditions.

Evaluation Metrics. For code generation tasks (HumanEval, LiveCodeBench) and mathematical reasoning benchmarks (GSM8K, AIME’24, AMC12’23), we use accuracy (or pass@1). In question answering, accuracy is used for GPQA, while MT-Bench scores are obtained using Qwen2.5-72BInstruct [11] as the judge. The accuracy on livecodebench is averaged over 8 samples while the accuracy on other datasets are averaged over 16 samples. We calculate the acceptance rate over the entire generation process and the accuracy of the final generated text. The evaluation procedure works

- as follows: at each generation step, we obtain outputs from both the draft and target models, then use a verifier to determine whether to accept or reject the draft output. The accepted result is added to the history trajectory, and this iterative process repeats until the end of generation is reached.

### 4.1 End-to-End Performance of LOOKAHEAD REASONING

We evaluated the end-to-end performance of LOOKAHEAD REASONING (LR) across diverse benchmarks using DeepSeek-R1-Distill and Qwen3 pairs. The detailed results are presented in Table 1. A key finding is LR’s consistent ability to preseve task accuracy. Across a variety of benchmarks, LR’s accuracy varies within a narrow range relative to the target model’s autoregressive baseline, from approximately 1.0% above to 2.1% below baseline performance. This accuracy preservation contrasts with SpecReason, which exhibited more noticeable accuracy reductions on several tasks (e.g., dropping from 91.8% to 85.9% on GSM8K with Deepseek-R1, a ∼ 6% decrease). This underscores LR’s design principle of preserving output via robust semantic verification.

Furthermore, LR achieves strong accuracy while maintaining high step acceptance rates, often above 50% and reaching up to 63%. These substantial acceptance rates empirically support our initial insight that a smaller draft model can effectively predict semantically correct reasoning steps for a larger target model. LR also delivers significant efficiency gains. Its step-level parallelism is orthogonal to token-level speculative decoding, and their synergy produces substantial speedups. LR alone achieves speedups ranging from 1.04x to 1.71x across various benchmarks and model pairs. When combined with n-gram SD, the total speedup is further amplified, reaching up to 2.11x. This combined approach consistently outperforms n-gram SD alone, demonstrating the added value of step-level speculation. These results, consistent across both Deepseek-R1 and Qwen3 families, underscore the generalizable acceleration benefits of LR.

### 4.2 Combining LOOKAHEAD REASONING with Speculative Decoding

To empirically validate the orthogonality of LOOKAHEAD REASONING (LR) with speculative decoding, we conducted experiments using prompt-lookup decoding (n-gram) on the AIME dataset.

- Figure 3 shows the orthogonality of LR and Speculative Decoding (SD). Subplot (a) shows that while LR alone with varying draft step number reaches a speedup around 1.4x, adding SD boosts this to approximately 1.9x. Similarly, subplot (b) illustrates that SD alone with varying Speculative Token Numbers peaks around 1.55x speedup, but combining it with LR again achieves up to 1.9×. Collectively, these results highlight that while either method in isolation offers limited gains, their

- Table 1: LOOKAHEAD REASONING’s Performance Across Datasets. Speedup is relative to the Autoregressive Decoding of the respective Target Model.

Dataset AIME24 AMC23 GSM8K HumanEval GPQA MT-Bench LiveCodeBench

Method Metric

Draft: Deepseek-R1-Distill 1.5B / Target: Deepseek-R1-Distill 32B

Draft Model Acc. (%) 28.5 ± 3.9 71.6 ± 4.1 77.6 ± 3.3 67.2 ± 2.4 9.6 ± 1.2 6.23 ± 1.9∗ 14.5 ± 1.3 Target Mode Acc. (%) 70.8 ± 5.2 95.6 ± 2.3 91.8 ± 1.9 96.9 ± 0.8 63.3 ± 2.2 8.17 ± 1.2∗ 48.9 ± 1.3 SpecReason Acc. (%) 58.3 ± 5.7 90.6 ± 2.6 85.9 ± 2.2 94.5 ± 1.5 57.0 ± 2.8 – 40.6 ± 1.5

Apt. 0.39 0.69 0.93 0.43 0.08 – 0.25

Acc. (%) 69.2 ± 8.1 94.1 ± 2.1 92.8 ± 1.8 95.5 ± 1.8 61.2 ± 2.8 8.13 ± 1.2∗ 49.5 ± 2.3 Apt. 0.47 0.58 0.63 0.44 0.35 0.48 0.47 Speedup 1.36× 1.48× 1.71× 1.27× 1.14× 1.27× 1.21× SD Speedup 1.53× 1.50× 1.39× 1.32× 1.48× 1.25× 1.45× SD+LR(ours) Speedup 1.82× 2.00× 2.11× 1.54× 1.63× 1.51× 1.58×

LR(ours)

###### Draft: Qwen3 1.5B / Target: Qwen3 32B

Draft Model Acc. (%) 46.9 ± 8.1 84.2 ± 4.7 91.1 ± 1.6 85.4 ± 1.6 38.5 ± 1.4 7.96 ± 1.5∗ 28.8 ± 1.6 Target Model Acc. (%) 80.0 ± 3.9 97.5 ± 2.0 96.6 ± 1.4 97.6 ± 0.8 68.2 ± 2.1 8.53 ± 1.1∗ 52.4 ± 1.4 SpecReason Acc. (%) 68.3 ± 5.3 90.5 ± 3.9 94.5 ± 1.4 92.0 ± 2.0 66.3 ± 2.0 – 39.7 ± 1.9

Apt. 0.75 0.92 0.95 0.91 0.46 – 0.65

Acc. (%) 80.4 ± 4.1 96.4 ± 2.0 96.4 ± 1.2 97.1 ± 0.8 68.5 ± 2.4 8.46 ± 1.15∗ 51.7 ± 1.7 Apt. 0.43 0.53 0.50 0.39 0.30 0.38 0.40 Speedup 1.12× 1.22× 1.32× 1.13× 1.04× 1.10× 1.08× SD Speedup 1.40× 1.38× 1.32× 1.32× 1.40× 1.41× 1.25× SD+LR(ours) Speedup 1.49× 1.62× 1.68× 1.39× 1.44× 1.49× 1.32×

LR(ours)

Note: LR (ours) refers to our proposed Lookahead Reasoning method. SD denotes token-level Speculative Decoding (N-gram based). Acc. stands for Accuracy (%), and Apt. for Acceptance Rate. For MT-Bench (marked with ∗), the reported metric is its standard score (0-9 scale) instead of accuracy. "–" indicates data not applicable. Speedup is relative to the autoregressive decoding of the respective target model.

combination consistently yields the most significant performance improvements, aligning with our theoretical analysis in § 3.2.

2.0

1.8

###### Speedup

LR Only

LR + SD

1.6

Combining Gain

1.4

1.2

5 10 15 20 25 30

Speculative Steps

(a) vary the draft step number for LR

2.0

1.8

###### Speedup

1.6

SD Only LR + SD Combining Gain

1.4

| |
|---|

1.2

5 10 15 20 25 30

Speculative Tokens

(b) vary the speculative token number for SD

Figure 3: Orthogonality of Lookahead Reasoning and Speculative Decoding. When used alone, the speedup from both LR and SD is limited by their draft length (γ). However, their combination consistently improves the max achievable speedup.

### 4.3 Ablation Study

Effectiveness of the Verifier. We conducted an ablation study to assess the impact of different verifier mechanisms on task accuracy, utilizing DeepSeek-R1-Distill 32B as the target model and a 1.5B parameter model as the draft model on GSM8K and AIME’24 datasets. We compare four verifiers: (1) Random Acceptance of drafts; (2) LLM-as-a-Judge (LLM-J) with Qwen2.5 7B/32B [11]; (3) Embedding-based Verification (Emb.) with all-mpnet-base-v2 model [8] at 0.85/0.95 similarity thresholds; and (4) Reasoning Model Scoring(Score) [9], using the target model to assign 0-9 scores with acceptance thresholds of 7/9. Results are in Table 2.

LLM-J verifiers (both 7B and 32B) showed robust accuracy preservation across both datasets, with minimal performance difference observed between the two verifier sizes. On GSM8K, their

performance closely aligned with the original baseline, indicating no accuracy degradation. On AIME, LLM-J verifiers also maintained accuracy very close to the original, with observed deviations within approximately 1-2%. This contrasts sharply with Random Acceptance. Despite comparable or lower acceptance rates (e.g., 0.50 on GSM8K and 0.40 on AIME), Random Acceptance led to significant accuracy degradation on both GSM8K (∼ 3.5% lower) and AIME ( ∼ 11% lower). This underscores the necessity of a robust verification mechanism.

The Embedding-based verifier shows a trade-off: the stricter 0.95 threshold on GSM8K preserved accuracy (92.3 ± 1.4%) at a lower acceptance rate (0.37), while the 0.85 threshold, despite a higher acceptance rate (0.56), resulted in a ∼ 2% accuracy drop (89.8 ± 2.5%). This pattern was mirrored on AIME. This indicates that while semantic equivalence is a promising criterion, its efficacy in preserving accuracy is highly dependent on the stringency (and precision) of the similarity judgement.

Reasoning Model Scoring, which assesses draft quality via target model scores rather than direct equivalence, consistently underperformed in accuracy preservation. For instance, even employing the stricter Threshold 9 resulted in notable accuracy reductions of approximately 5.9% on GSM8K and 12.5% on AIME. The still relatively high acceptance rate on GSM8K with this threshold (e.g., 0.93) suggests that the scoring mechanism may possess limited discriminative power on simpler datasets, even at stricter thresholds. This highlights a fundamental limitation: quality scores, even with high thresholds, do not reliably ensure alignment with the target model’s output distribution, which is critical for LOOKAHEAD REASONING’s correctness.

These results reveals that verifiers grounded in semantic equivalence with the target model’s likely output are most effective for preserving accuracy within LOOKAHEAD REASONING. LLM-as-a-Judge excels in this, provding nuanced judgement, though with potential computational overhead. Embedding models provide a lightweight alternative (e.g., all-mpnet-base-v2 is only ∼100M parameters), where performance is tunable via the similarity threshold, offering a cost-effective solution.

- Table 2: Performance comparison with different verifiers on GSM8K and AIME datasets. Apt.: accept rate; Acc.: accuracy (%).

LLM-J (Qwen) Emb. (Th.) Score (Th.) 7B 32B 0.85 0.95 7 9

Dataset Metric Orig. Rand.

Apt. − 0.50 0.63 0.58 0.56 0.37 0.97 0.93 Acc. 91.8 ± 1.9 88.3 ± 3.7 92.8 ± 1.8 92.3 ± 1.2 89.8 ± 2.5 92.3 ± 1.4 82.1 ± 2.4 85.9 ± 2.2

GSM8K

Apt. − 0.40 0.47 0.46 0.45 0.38 0.81 0.39 Acc. 70.8 ± 5.2 59.6 ± 5.4 69.2 ± 8.1 69.0 ± 4.7 64.0 ± 6.5 66.7 ± 6.3 37.9 ± 7.5 58.3 ± 5.7

AIME

Effect of Tree Width on Performance We investigate the impact of speculation tree width on LOOKAHEAD REASONING’s accuracy, accept rate, and speedup, keeping depth γ = 2. In LOOKAHEAD REASONING, candidate sequences grow as Wγ. Wider trees (W > 1) can boost accept rate but escalate FLOPs and, with imperfect verifiers, risk accuracy degradation due to erroneous acceptances. We hypothesize a stronger verifier mitigates this. Experiments on GSM8K and AIME24 used Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct as judges. Results are demonstrated in Table 3.

Increasing W consistently raised accept rate across datasets and judges (e.g., GSM8K with Qwen2.57B: accept rate 0.63 for W = 1 to 0.83 for W = 8). However, this rarely translated to better speedup beyond W = 2. Further widening often diminished speedup (e.g., AIME24 with Qwen2.57B, W = 8 yielded no speedup), likely due to the exponential overhead outweighing accept rate gains. Accuracy trends highlight verifier importance. With the Qwen2.5-7B judge, increasing W led to a noticeable accuracy drop, especially on AIME24 (from 69.2% at W = 1 to 64.6%

- at W = 8), supporting our hypothesis. The stronger Qwen2.5-32B judge demonstrated greater resilience: accuracy remained more stable on GSM8K, and the degradation on AIME24 was less pronounced (69.0% at W = 1 to 67.3% at W = 8). This indicates a stronger verifier is crucial for wider trees to manage the increased risk of incorrect speculation.

## 5 Related Work

Speculative Decoding. There are many different types of speculative decoding approaches. Drafthead methods like Medusa [23], Hydra [24], and EAGLE [25, 10, 13] integrate auxiliary heads into the target model to propose sequences. In contrast, Jacobi-based approaches such as Lookahead Decoding [12] and CLLM [26] enable parallel n-gram generation without draft models. Systemlevel efforts [27, 28] further optimize SD’s runtime efficiency in serving systems. LOOKAHEAD

Table 3: Impact of Tree Width (W) on Performance Metrics (Depth γ = 2)

W=1 W=2 W=4 W=8 Acc.(%) Apt. Spd. Acc.(%) Apt. Spd. Acc.(%) Apt. Spd. Acc.(%) Apt. Spd.

Dataset Judge

Qwen7B 92.8 ± 1.8 0.63 1.48× 91.2 ± 1.8 0.73 1.49× 91.1 ± 1.7 0.77 1.47× 91.5 ± 1.8 0.83 1.25× Qwen32B 92.3 ± 1.2 0.58 1.40× 93.2 ± 2.0 0.66 1.42× 92.8 ± 1.8 0.73 1.39× 92.5 ± 1.5 0.77 1.19×

GSM8K

Qwen7B 69.2 ± 8.1 0.47 1.27× 67.3 ± 4.1 0.58 1.32× 65.4 ± 6.5 0.67 1.26× 64.6 ± 5.9 0.74 1.00× Qwen32B 69.0 ± 4.7 0.46 1.23× 69.0 ± 6.7 0.54 1.23× 68.1 ± 6.1 0.59 1.17× 67.3 ± 7.1 0.68 0.98×

AIME24

Note: Acc.: Accuracy (%); Apt.: Accept Rate; Spd.: Speedup (relative to target model autoregressive decoding). Depth γ = 2. Qwen2.57B/32B-Instruct are judge models.

REASONING introduces a complementary form of step-level speculation tailored for reasoning models, enabling a new dimension of parallelism orthogonal to token-level methods.

LLM Reasoning. Recent trends shift from scaling model size [29, 30] to scaling inference-time compute [31, 32, 33], enabling large reasoning models (LRMs) like OpenAI o3/o4 [5], KimiK1.5 [34], and DeepSeek-R1 [4] to generate longer CoT to solve more complex problems in many steps. Recent work has begun to leverage this inherent step-by-step structure to accelerate LRMs. For instance, Speculative Thinking [35] uses LRMs to guide a smaller draft model, while SpecReason [9] accelerates reasoning by employing the large model to score the output of a small model, thereby deciding whether to accept its generated step. However, unlike LOOKAHEAD REASONING, these methods do not pursue a step-level equivalent with the original target model to accelerate reasoning.

## 6 Limitation and Conclusion

This paper introduces LOOKAHEAD REASONING, a novel method for accelerating large reasoning models during long CoT reasoning. LOOKAHEAD REASONING adds a new step-level dimension of parallelism, complementing traditional token-level speculative decoding. Our approach uses a draft model to propose future reasoning steps, which are then semantically verified by the target model. Evaluation on various datasets using two open-source reasoning models show that it can achieve up to 2.1X speedup combined with speculative decoding. This highlights LOOKAHEAD REASONING’s effectiveness in making large reasoning models faster. This work has limitations that suggest future improvements. First, using ‘\n\n‘ to split reasoning steps is simple but may miss optimal breaks; smarter segmentation is needed. Second, current verifiers trade speed for accuracy; faster, lightweight alternatives remain an open challenge.

## References

- [1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [2] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024.
- [3] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.
- [4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [5] OpenAI. Openai o3 and o4-mini system card. April 2025.

- [6] Qwen Team. Qwen3 technical report. 2025.
- [7] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.
- [8] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019.
- [9] Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. Specreason: Fast and accurate inference-time compute via speculative reasoning. arXiv preprint arXiv:2504.07891, 2025.
- [10] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. arXiv preprint arXiv:2406.16858, 2024.
- [11] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [12] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. arXiv preprint arXiv:2402.02057, 2024.
- [13] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840, 2025.
- [14] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.
- [15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [16] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [17] AIME. Aime problems and solutions, 2025.
- [18] AMC12. Amc 12 problems and solutions, 2025.
- [19] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.
- [20] Qwen Team. Qwen2.5-math: The world’s leading open-sourced mathematical llms, 2024.
- [21] Apoorv Saxena. Prompt lookup decoding, November 2023.
- [22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

- [23] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.
- [24] Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. Hydra: Sequentially-dependent draft heads for medusa decoding. arXiv preprint arXiv:2402.05109, 2024.
- [25] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024.
- [26] Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. In Forty-first International Conference on Machine Learning, 2024.
- [27] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating generative large language model serving with tree-based speculative inference and verification. arXiv preprint arXiv:2305.09781, 2023.
- [28] Xiaoxuan Liu, Cade Daniel, Langxiang Hu, Woosuk Kwon, Zhuohan Li, Xiangxi Mo, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Optimizing speculative decoding for serving large language models using goodput. arXiv preprint arXiv:2406.14066, 2024.
- [29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [30] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [31] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [32] Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, and Zaid Harchaoui. From decoding to meta-generation: Inference-time algorithms for large language models. arXiv preprint arXiv:2406.16838, 2024.
- [33] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [34] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [35] Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. Speculative thinking: Enhancing small-model reasoning with large model guidance at inference time. arXiv preprint arXiv:2504.12329, 2025.

## A Judgement Prompt Template

Semantic Equivalence Analysis Prompt

<|im_start|>system You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|> <|im_start|>user Evaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning. Focus on semantic similarity rather than exact wording. Compare the main ideas, key points, overall message, logical structure, and numerical calculations/results of both reasoning steps. If the reasoning steps convey essentially the same meaning and generate same calculation results, respond with [aligned]. If the reasoning steps express different meanings, respond with [unaligned]. If it is too hard to determine, respond with [unaligned] Please directly provide the final result in [aligned] or [unaligned].

- Reasoning step 1 (s1):

- <start_s1> {}

- <end_s1>

Reasoning step 2 (s2): <start_s2> {}

- <end_s2><|im_end|> <|im_start|>assistant [

This prompt template is specifically designed for Qwen2.5 Instruct models. It guides the model to directly output either "aligned" or "unaligned" as its judgment. Consequently, a user can determine semantic equivalence between two sentences (s1 and s2) by simply checking if the model’s initial output string begins with "ali" (the first three letters of "aligned"). Thus, only one forward pass is needed to get the result and judgement latency can be largely saved.

## B Detailed Speedup Analysis B.1 Performace Gains Analysis

### B.1.1 Speedup Analysis of Async Lookahead Reasoning

For the analysis that follows, we assume all sequences are of equal length and that the draft tree contains exactly one branch at each layer. Moreover, we treat the verifier’s overhead as negligible.

### Notation.

- • γ ∈ N: maximum number of generations the large model performs in parallel.
- • T: cost (wall-time) of one sentence generated by target-model.
- • c1: cost of one draft-model run, measured in units of T (so drafting costs c1T).
- • α1 ∈ (0,1):accept rate of the drafts
- • Xi: number of consecutive accepted drafts in stage i before the first rejection.

We view a full generation as a sequence of DRAFT STAGE, each of which proceeds as follows:

- 1. If the number of generations the large model performs in parallel is less than γ. The draft model sequentially generate drafts.
- 2. Each time when we start to generate a draft step, we immediately ask the target model to generate a target step.
- 3. After the target model finished generation, immediately ask the verifier to verify whether should we accept the draft. If the draft was reject, fall back to the target model’s original sentence and proceed to the next DRAFT STAGE.

Since each draft is accepted independently,

P(Xi = k) = αk(1 − α), E[Xi] =

α 1 − α

.

- Theorem 1. The latency speedup for sync Lookahead Reasoning is

fsync(γ) =

1 − αγ (1 − α)(1 − c + cγ)

.

Proof. The proof follows the same reasoning as in [3]. The only difference is that our γ represents the maximum number of tokens the large model generates in parallel, whereas in their notation, it corresponds to γ + 1

| |
|---|

- Theorem 2. Let

total sentences generated by our algorithm in n DRAFT STAGE total sentences generated by target-only model in n DRAFT STAGE

S =

,

We can see this as the latency speedup using async Lookahead Reasoning algorithm. Then:

#### 1. If γ ≥ ⌈c1

⌉, the draft tree never saturates. The parallel dimension of the target model is ⌈c1

1

⌉, and as n → ∞, the asymptotic speedup is

1

1 c1 + (1 − c1)(1 − α1)

S1 =

.

#### 2. If γ < ⌈c1

⌉, the draft tree is depth-limited. The parallel dimension of the target model is γ, and as n → ∞, the asymptotic speedup is

1

1 − α1γ (1 − α1) + c1 α1 − α1γ+1 − γ(1 − α1)α1γ

S2 =

.

Proof. Over n stages, we compare the total number of sentences generated by our algorithm to that produced by the baseline (target-only) approach.

### Case 1 When γ ≥ c 1

, note that since each draft costs c1T, the draft model can generate at most

1

1 c1 sentences during the time T required for the target model to produce one sentence. Therefore,

. Moreover, in DRAFT STAGE i our algorithm spends

the draft tree never saturates, and the parallel dimension of the target model is effectively c 1

1

#### T + c1T Xi

total time. Over that same interval, the baseline target-only model would have produced

T + c1T Xi T

= 1 + c1 Xi

sentences, while our algorithm emits 1 + Xi sentences. Thus over n stages, according to the Law of Large Numbers,

(1 + Xi) i 1 + c1Xi

- S1(n) = i

=

n + i Xi n + c1 i Xi −−−−→

n→∞

1 + E[Xi] 1 + c1 E[Xi]

=

1 c1 + (1 − c1)(1 − α1)

.

### Case 2 When γ < ⌈c1

⌉, the draft-tree saturates at depth γ. So the parallel dimension of the target model would be γ To emit a total of Xi + 1 sentences in stage i, we therefore proceed in

1

Xi+1 γ

full intervals of length T, plus a final partial batch of size Xi mod γ. Hence the total wall-time for stage i is

Xi+1

γ T + c1T Xi mod γ .

Over that same interval, the baseline target-only model would have generated

Xi+1

γ + c1 Xi mod γ

sentences, whereas our algorithm emits 1 + Xi. Therefore, as n → ∞

n + ni=1 Xi

1 − α1γ (1 − α1) + c1 α1 − α1γ+1 − γ(1 − α1)α1γ

γ + c1 ni=1(Xi mod γ) −→

- S2(n) =

.

n i=1

Xi+1

We put the calculation details in the appendix.

| |
|---|

### B.1.2 Optimal Speculation Strategies under Concurrency Constraints

In this section we show that under a fixed parallelism budget, the optimal inference speedup is always achieved by jointly applying step-level and token-level parallelism, rather than by using either in isolation. Concretely, let γ2 denote the degree of parallelism used by speculative decoding and c2 be the ratio of the per-step execution time of the draft model to that of the target model. Then token-level speculative decoding alone yields[3]

1 − αγ

2

2

. (2)

g(γ2) =

(1 − α2)(1 − c2 + c2 γ2)

(Note: here the formula is a little different than the one in [3] due to different definition.) Next, a pure step-level asynchronous parallelism scheme of depth γ1 achieves speedup

 

1 − αγ

1

1 1 − α1 + c1 α1 (1 − αγ

, γ1 < ⌈c1

⌉ S1 =

S2 =

1 ) − c1 γ1 αγ

1 (1 − α1)

1

1

1

fasync(γ1) =

1 c1 + (1 − c1)(1 − α1)



, otherwise

and a pure step-level synchronous parallelism scheme of depth γ1 achieves speedup

1 − αγ

1

1

. (3) When both schemes are combined, the resulting speedup factorizes:

fsync(γ1) =

(1 − α1)(1 − c1 + c1 γ1)

f(γ1) × g(γ2). (4) Since in real-world systems, due to memory or compute constraints, we often need to cap the total degree of parallelism i.e.γ1γ2 in a hybrid speculative setting to M. In this case, we may ask that given a finite parallel budget M, is it better that we combine these two parallel dimensions or we only use one of them? Thus our design goal becomes

h(γ1,γ2) = f(γ1) × g(γ2). (5) Where ParallelDimg = γ2, and

max

ParallelDimg×ParallelDimf≤M

γ1, sync min{⌈1c⌉,γ1}, async

ParallelDimf =

It’s easy to see that if we set γ1 = 1, then we are using purely token-level speculative decoding, whereas if we set γ2 = 1, then we are using purely Lookahead reasoning.

- Theorem 3. Under synchronous Lookahead Reasoning with concurrency budget M ≥ 4, M is an even number, and the mild parameter constraint

1 + α1 1 + c1

1 + α2 1 + c2

> 1.157, (6) Then at least one of the following must hold:

min

,

- 1 + α1 1 + c1 ≥

(1 + α2M/2)(1 − c2 + c

2M

2 ) 1 − c2 + c2M

, (7)

- 1 + α2 1 + c2 ≥

(1 + α1M/2)(1 − c1 + c

- 1M

- 2 ) 1 − c1 + c1M

. (8)

Furthermore, if both (7) and (8) hold simultaneously, then combining both speculative techniques strictly outperforms using either one alone. Conversely:

- • If (7) fails, the optimal strategy is to use only token-level speculation.
- • If (8) fails, the optimal strategy is to use only step-level speculation.

### Proof. Step 1: At least one of (7) and (8) must hold Define, for i = 1,2,

1 − ci + c

iM 2

Di(M) = 1 + αiM/2

. Observe that both factors

1 − ci + ciM

1 − ci + c

iM 2

1 + αiM/2 and

1 − ci + ciM

are strictly decreasing in M. Hence each Di(M) decreases as M grows. In particular,

1 + αi 1 + ci

Di(2) =

.

Since either D1(2) ≥ D2(2) or D1(2) < D2(2), it follows that either

1 + α1 1 + c1

D1(2) > D2(M) =⇒

> D2(M), or

1 + α2 1 + c2

D2(2) > D1(M) =⇒

> D1(M). In other words, at least one of (7) or (8) must hold.

- Step 2: Token-level-only speculation is suboptimal if condition (7) holds; otherwise, it is the optimal strategy. Lemma 3 shows that both fsync(γ1) and g(γ2) are unimodal, reaching their maxima at γ1∗ ≥ 2 and γ2∗ ≥ 2. Below, we analyze whether token-level-only speculation (i.e. γ1 = 1) yields the best speedup for different ranges of the concurrency budget M.

- Case 1: If M < γ2∗ In this case, when (7) holds, we have h(1,M) ≤ h(2, M2 )(transform through formulas), then according to the monotonicity of h, we have

M 2

h(1,γ2) ≤ h(1,M) ≤ h(2,

), ∀1 ≤ γ2 ≤ M

Therefore, the overall maximum is attained only by jointly employing both levels. However, when (7) fails, since Di(x) is strictly decreasing, we know that

(1 + α2x/2)(1 − c2 + c

2x 2 )

1 + α1 1 + c1

1 − c2 + c2x ∀x ≤ M In this case, according to Lemma 5,

<

h(γ1,γ2) ≤ h(1,γ1γ2) ≤ h(1,M), ∀γ1,γ2 ≥ 1,γ1γ2 ≤ M

So the overall maximum is attained when we use token-level-only technique.

- Case 2: If γ2∗ ≤ M < 2γ2∗ In this case, we first prove that

h(1,γ2∗) < h(2,

γ2∗ 2

), ∀1 ≤ γ2 ≤ M Then we prove that 7 is always hold. From above we can see

h(1,γ2) ≤ h(1,γ2∗) < h(2,

γ2∗ 2

), ∀1 ≤ γ2 ≤ M And the theorem follows.

- 1.From Lemma3, we can know that

a2(γ2∗) =

αγ

∗ 2

2 (−lnαγ

∗ 2

2 ) 1 − αγ

∗

2

=

c2γ2∗ 1 − c2 + c2 γ∗

2

Therefore

h(2,

γ2∗ 2

)/h(1,γ2∗) =

1 + α1 1 + c1

- 1 − c2 + c2γ2∗

(1 + αγ

∗

- 2 /2

2 )(1 − c2 + c2γ

∗ 2

2 )

=

1 + α1 1 + c1

- 1

1 + αγ

∗ 2 /2

- 2

1 1 − 12a2(γ∗

2) Let y = 1 − αγ

∗ 2 /2

2 ∈ (0,1), (1 + αγ

∗ 2 /2

2 )(1 −

- 1

- 2

a2(γ2∗)) = 2 − y + (y − 2 +

1 y

)ln(1 − y)

Then from Lemma 6, we can see that its value was less than 1.157. Then given 6 we can have

h(2,

γ2∗ 2

)/h(1,γ2∗) > 1

- 2.From previous step, we know that D2(M) is strictly decreasing, so

D2(M) ≤ D2(γ2∗) < 1.157 <

1 + α1 1 + c1

So 7 is always hold.

- Case 3: If M ≥ 2γ2∗ In this case, same as in Case 2, 7 is always hold. Besides, we have h(1,γ2) ≤ h(1,γ2∗) < h(2,γ2∗), ∀1 ≤ γ2 ≤ M

So it’s obvious token-level-only speculation is suboptimal.

- Step 3: If (8) holds, step-level-only speculation is suboptimal. Otherwise, it is the optimal strategy. Same as the previous step.

| |
|---|

- Theorem 4. Under asynchronous Lookahead Reasoning with 0.52 < α1,α2 < 0.8 and c1 < 1 3,c2 < 51 and the total parallelism budget M ≥ 16 and M is an even number. Then under constraint

⌉,γ1} × γ2 ≤ Mthe overall speedup h(γ1,γ2) is maximized if and only if both step-level and token-level parallelism are employed (i.e. γ1 ≥ 2 and γ2 ≥ 2).

min{⌈c1

1

Proof. Step 1: Token-level-only is not optimal. Lemma 3 shows that g(γ2) is unimodal and reaching their maxima at γ2∗ ≥ 2.

- 1.M ≥ 2γ2∗ This case, we have h(1,γ2) ≤ h(1,γ2∗) ≤ h(2,γ2∗), ∀1 ≤ γ2 ≤ M

Therefore, it’s obvious that token-level-only is not optimal.

- 2.γ2∗ ≤ M < 2γ2∗ In this case, we only need to prove

h(1,γ2∗) ≤ h(2,

γ2∗ 2

) Same as the proof in 3, we only need to prove that

1 + α1 1 + c1 ≥ 1.157

is hold, and it’s easy to prove.

- 3.M < γ2∗ This case, according to Lemma 1, we can see that

γ2 2

h(1,γ2) < h(2,

)

is always hold, so we can conclude that tokne-level-only is not optimal.

Step 2: Step-level-only is not optimal. From Lemma 2, we can know that fasync(γ1) is strictly

⌉, and will stay constant after that.

increasing when 1 ≤ γ ≤ ⌈c1

1

#### 1. M ≤ ⌈c1

⌉ This case, according to Lemma 1,

1

M 2

,2), ∀γ1 ≤ M So step-level-only is not optimal.

h(γ1,1) ≤ h(M,1) < h(

#### 2.⌈c1

⌉ This case, according to Lemma 1

⌉ < M < 2⌈c1

1

1

1 c1⌉,1) = h(M,1) < h(

M 2

h(γ1,1) ≤ h(⌈

,2), ∀γ1 ≤ M

#### 3. M ≥ 2⌈c1

⌉ This case, we will have

1

1 c1⌉,1) < h(⌈

1 c1⌉,2), ∀1 ≤ γ1 ≤ M

h(γ1,1) ≤ h(⌈

So step-level-only is not optimal.

- Lemma 1. Let w(γ) = f(γ) g M/γ ,

where M is an even number, and

| |
|---|

- • f(γ) =

1 − α1γ 1 − α1 + c1 α1(1 − α1γ) − c1 γ α1γ(1 − α1)

,

- • g(γ) =

1 − α2γ (1 − α2)(1 − c2 + c2 γ)

,

- • 0.5 < α1,α2 < 0.8, 0 < c1 < 13,0 < c2 < 15 and M ≥ 16.

Then w(2) > w(1), w(M2 ) > w(M). Proof.

1 − c2 + c2M 1 − c2 + c2M/2 >

(1 + α1) (1 + α2M/2)(1 + α1c1(1 − α1))

w(2)/w(1) =

(1 + α1) (1 + α2M/2)(1 + α1c1(1 − α1))

× 1

We can easily found that this function is monotonically decreasing with respect to c1, α2 M,monotonically increasing with respect to α1. Therefore,

1 + 0.52 (1 + 0.88)(1 + 0.52 × 1/3 × 0.48)

w(2)/w(1) >

> 1

1 − α1 + c1α1(1 − α1M) − c1Mα1M(1 − α1) (1 + α1M/2)(1 − α1 + c1α1(1 − α1M/2) − c1M2 α1M/2(1 − α1)

M 2

1 + α2 1 + c2

w(

)/w(M) =

1 − α1M/2 2

α1M/2 1 + α1M/2

1 − α1 (1 − α1)(1 + c1(α1 + ··· + α1M/2) − c1M2 α1M/2)

1 + α2 1 + c2

[1 − (1 − c1M

=

)

]

Given that α1k ≥ α1M/2,k ∈ {1,2,··· ,M/2}, so we have

M 2

α1M/2 > 0 If

c1(α1 + α12 + ··· + α1M/2) − c1

1 − α1M/2

2 ≤ 0 then

1 − c1M

1 + α2 1 + c2

M 2

)/w(M) ≥

> 1 Otherwise,

w(

α1M/2 1 + α1M/2

0.88 1 + 0.88

1 + 0.52 1 + 1/5

1 + α2 1 + c2

M 2

(1 −

× 1] >

[1 − 1 ×

)/w(M) >

) > 1

w(

| |
|---|

- Lemma 2. The speedup function of the async Lookahead Reasoning

 

1 − α1γ 1 − α1 + c1 α1 (1 − α1γ) − c1 γ α1γ (1 − α1)

, γ < ⌈c1

⌉ S1 =

S2 =

1

fasync(γ) =

1 c1 + (1 − c1)(1 − α1)



, otherwise

is strictly increasing when 1 ≤ γ ≤ ⌈c1

⌉,γ ∈ N+,and will stay constant after γ ≥ ⌈c1

⌉,γ ∈ N+

1

1

### Proof. We first see the function as a continuous function on R and prove that when 1 ≤ γ < ⌈c1

⌉,

1

the speedup function is strictly increasing Write

A(γ) = 1 − α1γ, D(γ) = 1 − α1 + c1 α1 (1 − α1γ) − c1 γ α1γ (1 − α1). Then fasync(γ) = A(γ)/D(γ), and by the quotient rule

A′(γ)D(γ) − A(γ)D′(γ) D(γ)2

fasync′ (γ) =

. We compute

A′(γ) = −α1γ lnα1, D′(γ) = −c1 α1γ α1 lnα1 + (1 − α1) 1 + γ lnα1 . Hence the numerator of fasync′ (γ) becomes A′(γ)D(γ) − A(γ)D′(γ) = α1γ (−lnα1)D(γ) + c1 (1 − α1γ) α1 lnα + (1 − α1)(1 + γ lnα1)

= α1γ(1 − α1) (−lnα1) + c1 γ α1γ (lnα1) + c1 (1 − α1γ)(1 + γ lnα1)

= α1γ(1 − α1) (−lnα1)(1 − c1γ) + c1(1 − α1γ)

Since

−lnα1 > 0, 1 − α1 > 0, 1 − c1γ > 0, 1 − α1γ > 0 each term is strictly positive for all γ ≥ 1. Therefore

A′(γ)D(γ) − A(γ)D′(γ) > 0 =⇒ fasync′ (γ) > 0,

Then we prove that S1 ≥ S2 We only need to prove that when γ = c1

, we have S2 = S1 When γ = c1

1

,

1

1 − α1γ 1 − α1 + c1 α1 (1 − α1γ) − c1 γ α1γ (1 − α1)

S2 =

1 − α1γ 1 − α1 + c1 α1 (1 − α1γ) − α1γ (1 − α1)

=

1 − α1γ (1 − α1)(1 − α1γ) + c1 α1 (1 − α1γ)

1 c1 + (1 − c1)(1 − α1)

= S1 Therefore, So the lemma follows.

=

=

| |
|---|

- Lemma 3. The speedup function of token-level speculative decoding[3] and is a unimodality function.

1 − α2γ (1 − α2)(1 − c2 + c2 γ)

,γ ∈ R

g(γ) =

where α2 > c2 Then g increases on [1,γˆ) and decreases on (ˆγ,∞) for a unique γˆ ∈ R+. In practical scenarios, where γ ∈ N+, the maximum of g(γ) is attained at some integer point γ∗ ∈ N+,γ∗ ≥ 2 Sync Lookahead Reasoning has a similar form, so it also has this property.

### Proof. Step 1. Compute g′(γ). Set

N(γ) = 1 − α2γ, D(γ) = (1 − α2) 1 − c2 + c2 γ , so that g(γ) = N(γ)/D(γ). By the quotient rule,

N′(γ)D(γ) − N(γ)D′(γ) D(γ)2

g′(γ) =

. Since

N′(γ) = −α2γ lnα2, D′(γ) = (1 − α2)c2, g′(γ) =

α2γ (−lnα2)(1 − α2)(1 − c2 + c2 γ) − c2 (1 − α2γ)(1 − α2) [(1 − α2)(1 − c2 + c2 γ)]2

. Since (1 − α2) > 0, the sign of g′(γ) equals the sign of

F(γ) := α2γ (−lnα2)(1 − c2 + c2 γ) − c2 (1 − α2γ).

### Step 2. F is strictly decreasing. Differentiate F:

F′(γ) = −α2γ (lnα2)2 (1 − c2 + c2 γ). Since 1 − c2 + c2γ > 0, it follows

F′(γ) < 0 for all γ ≥ 1. Thus F is strictly decreasing on [1,∞).

### Step 3. Signs of F at the ends.

- • At γ = 1: F(1) = α2(−lnα2) − c2(1 − α2)

since α2 > c2 > 0,−lnα2 > 1 − α2 > 0, so we have F(1) > 0

- • As γ → ∞, α2γ → 0, so F(γ) = α2γ (−lnα)(1 − c2 + c2 γ) + c − c2 (1 − α2γ) −→ −c < 0.

### Step 4. Conclusion via the Intermediate Value Theorem. Because F is continuous, strictly

decreasing, F(1) > 0, and limγ→∞ F(γ) < 0, there exists a unique γˆ ∈ R,γˆ ≥ 1 such that F(ˆγ) = 0. Moreover,

F(γ) > 0 ⇔ 1 ≤ γ < γ,ˆ F(γ) < 0 ⇔ γ > γ.ˆ Since g′(γ) and F(γ)) share the same sign, it follows that

g′(γ) > 0 for 1 ≤ γ < γ,ˆ g′(γ) < 0 for γ > γ,ˆ Besides, noting that

1 + α2 1 + c2

g(2)/g(1) =

so in practical scenarios where γ ∈ N+,we conclude that the maximum is achieved at some integer point γ∗ ≥ 2 as claimed.

| |
|---|

- Lemma 4. Let 0.5 < α < 0.8 and define

xαx 1 − αx

, x ≥ 1. Then:

a(α,x) = −lnα

- 1. For each fixed α ∈ (0.5,0.8), the function x  → a(α,x) is strictly decreasing on [1,∞).
- 2. For each fixed x ≥ 1, the function α  → a(α,x) is strictly increasing on (0.5,0.8).
- 3. Consequently, for every x ≥ 10 and α ∈ (0.5,0.8),

10 · 0.810 1 − 0.810 ≈ 0.26,

a(α,x) < a(0.8,10) = −ln(0.8)

and for all α ∈ (0.52,0.8),

a(α,2) ∈ a(0.52,2), a(0.8,2) ≈ (0.48, 0.79).

### Proof. (i) Monotonicity in x. Fix α ∈ (0,1) and write

xαx 1 − αx

N(x) D(x)

, N(x) = xαx, D(x) = 1 − αx. Then

=

f(x) =

N′(x) = αx 1 + xlnα , D′(x) = −αx lnα, and by the quotient rule

αx (1 + xlnα)(1 − αx) − x(−lnα)αx (1 − αx)2

N′(x)D(x) − N(x)D′(x) D(x)2

f′(x) =

. Since 0 < α < 1, setting u = xlnα < 0 we have by convexity of the exponential,

=

αx = eu > 1 + u = 1 + xlnα, hence

(1 + xlnα)(1 − αx) − x(−lnα)αx = 1 + xlnα − αx < 0. Thus f′(x) < 0. Because −lnα > 0, it follows immediately

∂ ∂x

a(α,x) = −lnα f′(x) < 0, so a(α,x) is strictly decreasing in x ≥ 1.

### (ii) Monotonicity in α. Fix x ≥ 1 and set

xαx 1 − αx

U(α) = −lnα, V (α) =

, so a(α,x) = U(α)V (α). Then

x2 αx−1 (1 − αx)2

1 α

U′(α) = −

, V ′(α) =

> 0. Hence

∂a ∂α

V (α) α

+ (−lnα)V ′(α). We claim this is > 0. Indeed,

= U′(α)V (α) + U(α)V ′(α) = −

V α

+ (−lnα)V ′ > 0 ⇐⇒ (−lnα)α V ′ > V. Since

−

x2αx−1 (1 − αx)2

x2 αx (1 − αx)2

xαx 1 − αx

α V ′ = α

, this inequality becomes

=

, V =

x2 αx (1 − αx)2

xαx

1 − αx ⇐⇒ x(−lnα) > 1 − αx. But for 0.5 < α < 0.8, the well-known bound −lnα > 1 − α and αx ≤ α imply

(−lnα)

>

x(−lnα) ≥ −lnα > 1 − α ≥ 1 − αx, so ∂a/∂α > 0. Thus a(α,x) is strictly increasing in α ∈ (0.5,0.8).

### (iii) Numerical bounds. By (i), a(α,x) ≤ a(α,10) for all x ≥ 10, and by (ii),

10 · 0.810 1 − 0.810 ≈ 0.26.

a(α,10) ≤ a(0.8,10) = −ln(0.8)

10 · 0.86 1 − 0.86 ≈ 0.48.

a(α,6) ≤ a(0.8,6) = −ln(0.8)

10 · 0.88 1 − 0.88 ≈ 0.36.

a(α,8) ≤ a(0.8,8) = −ln(0.8)

Also by (ii), for any α ∈ (0.52,0.8),

a(α,2) ∈ a(0.52,2), a(0.8,2) ∈ (0.48,0.8). This completes the proof.

| |
|---|

- Lemma 5. Let w(γ) = fsync(γ) g M/γ ,

here γ ∈ N+,M/γ ∈ N+ and assume M ≥ 4. Then:

- (a) If both (7) and (8) hold, then w(γ) is unimodal on [1,M] and attains its maximum at some γ∗ ∈ [2, M2 ].

- (b) If (7) fails, then the unique maximizer is γ = 1 (token-level only).i.e. h(γ1,γ2) < h(1,M),∀γ1γ2 = M,γ1,γ2 ∈ N+
- (c) If (8) fails, then unique maximizer is γ = M (step-level only). i.e. h(γ1,γ2) < h(M,1),∀γ1γ2 = M,γ1,γ2 ∈ N+

Proof. We first treat this function as a continuous function over R, analyze its derivative to determine its monotonicity, and then restrict its domain to N+ to obtain the desired results.

- Step 1: Derivative. By the product and chain rules,

w′(γ) =

1 γ

fsync(γ)g M/γ γ

fsync′ (γ) fsync(γ) −

M γ

g′ M/γ g M/γ

.

- Step 2: Log-derivatives. Define

ai(x) = x −ln(αi)αix 1 − αix

, i = 1,2. Then

γ

fsync′ (γ) fsync(γ)

= a1(γ) −

c1γ 1 − c1 + c1γ

, γ

g′(γ) g(γ)

= a2(γ) −

c2γ 1 − c2 + c2γ

. Hence with η = M/γ,

w′(γ) =

fsync(γ)g(η) γ

a1(γ) − a2(η) − c

1γ

1−c1+c1γ − c

2η

1−c2+c2η .

- Step 3: Monotonicity. By Lemma 4, a1(γ) is strictly decreasing in γ and a2(M/γ) strictly increasing. Also

c1γ 1 − c1 + c1γ −

c2η 1 − c2 + c2η

= 1 −

1 − c1 1 − c1 + c1γ −

c2 M (1 − c2)γ + c2M is strictly increasing in γ. Therefore w′(γ) is strictly decreasing on [1,M], and so w(γ) either strictly increasing, or strictly decreasing, or first increasing then decreasing on [1,M].

- Step 4: Endpoint comparison. Now we restrict the domain to be N+

- • If (7) fails, then w(2) < w(1), so w is either strictly decreasing on [1,M], or first increasing then decreasing and the critical points was in (1,2). Therefore we can conclude that w is strictly decreasing on [2,M]. Therefore we can conclude that there’s unique maximize at γ = 1, so we have

h(γ1,γ2) < h(1,M),γ1γ2 = M,γ1,γ2 ∈ N+

- • If (8) fails, then w(M2 ) < w(M), so w is either strictly increasing on [1,M], or first increasing then decreasing and the critical points was in (M2 ,M). Therefore we can conclude that w is strictly increasing on [1, M2 ]. Therefore we can conclude that there’s unique maximize at γ = M, so we have

h(γ1,γ2) < h(M,1),γ1γ2 = M,γ1,γ2 ∈ N+

- • If both (7) and (8) holds, then w(2) ≥ w(1),w(M2 ) ≥ w(M), so w would achieve the max between [2, M2 ]

So the lemma follows.

### Lemma 6. When 0 < y < 1

F(y) = 2 − y + (y − 2 +

1 y

)ln(1 − y) < 1.157

| |
|---|

Proof. Define

F(y) = 2 − y + y − 2 + y−1 ln(1 − y), 0 < y < 1. First derivative. A direct calculation gives

d dy

F′(y) =

(y − 2 + y−1)ln(1 − y) − 1 = (1 − y−2) ln(1 − y) −

1 y

.

### Second derivative and concavity of F′. Differentiating again,

d dy

d dy −y−1

F′′(y) =

(1 − y−2) ln(1 − y) +

1 − y−2 1 − y

y(1 + y) + 2ln(1 − y) y3

2 ln(1 − y) y3 −

1 y2

. Set

=

=

+

N(y) = y(1 + y) + 2ln(1 − y), so that F′′(y) = N(y)/y3. On (0,1),

d dy

2 1 − y

N′(y) =

y + y2 + 2ln(1 − y) = 1 + 2y −

and N(0) = 0. Hence N(y) < 0 for all y ∈ (0,1), which implies

F′′(y) < 0 on (0,1).

2y2 − y + 1 1 − y

= −

< 0,

Thus F′ is strictly decreasing. Sign-change of F′. - As y → 0+, ln(1 − y) ∼ −y, so

F′(y) ∼ (−y) 1 − y−2 − y1 = +O(y) > 0.

- As y → 1−, ln(1 − y) → −∞ while (1 − 1/y2) → 1, so F′(y) → −∞. By continuity and strict decrease, there is a unique y∗ ∈ (0,1) with F′(y∗) = 0, and

F′(y) > 0 (0 < y < y∗), F′(y) < 0 (y∗ < y < 1). Hence F increases on (0,y∗) and decreases on (y∗,1), so its maximum on (0,1) occurs at y∗. Numerical evaluation. Numerically one finds

y∗ ≈ 0.5693971022, F(y∗) ≈ 1.1562281731 < 1.157. Conclusion. Therefore for all y ∈ (0,1),

F(y) ≤ F(y∗) < 1.157, as claimed.

| |
|---|

## C Calculation of S2(n)

We’ll prove when n → ∞

n + ni=1 Xi

γ + c1 ni=1(Xi mod γ) −→

S2(n) =

n i=1

Xi+1

1 − α1γ (1 − α)1 + c1 α1 − α1γ+1 − γ(1 − α1)α1γ

.

### Step 1: Use the Law of Large Numbers According to the Law of Large Numbers

1 + n1 ni=1 Xi

n i=1(Xi mod γ) −→

S2(n) =

n i=1

γ + c

Xi+1

1 n

1

n

1 + E[Xi] E X

i+1 γ + c1E[Xi mod γ]

Here, the PMF of Xi is:

P(Xi = k) = α1k(1 − α1) And its expectation is:

α1 1 − α1

E[Xi] =

### Step 2: Compute E X

i+1 γ Let Y = X

i+1 γ .

Key Observation The ceiling function Xi+1

γ can be expressed in terms of integer thresholds. For m ≥ 0:

Xi + 1 γ

= m + 1 if Xi ∈ [mγ,(m + 1)γ − 1]

Thus:

Y = m + 1 for Xi ∈ [mγ,(m + 1)γ − 1], m = 0,1,2,... Compute E[Y ]

∞

(m + 1)P (mγ ≤ Xi ≤ (m + 1)γ − 1)

E[Y ] =

m=0

The probability P (mγ ≤ Xi ≤ (m + 1)γ − 1) is:

(m+1)γ−1

P(Xi = k) = α1mγ(1 − α1γ)

k=mγ

Therefore we can get:

Xi + 1 γ

1 1 − α1γ

E

=

- Step 3: Compute E [Xi mod γ] To calculate the expectation of Xi mod γ, where Xi follows the given geometric distribution and γ is an integer greater than 4, we proceed as follows: Compute P(Xi mod γ = r) For r ∈ {0,1,...,γ − 1}, we have:

∞

(1 − α1)α1r 1 − α1γ

P(Xi mod γ = r) =

P(Xi = mγ + r) =

m=0

Compute E[Xi mod γ] The expectation is:

γ−1

γ−1

α1 − α1γ+1 − γ(1 − α1)α1γ (1 − α1)(1 − α1γ)

(1 − α1)α1r 1 − α1γ

r · P(Xi mod γ = r) =

r ·

E[Xi mod γ] =

=

r=0

r=0

## D Illustration of Hybrid Approach

Token-Level Speculation

Accept draft Reject draft Target Prefix

Therefore, we have b*c = 468. Wait,

Therefore, we have b*c = 468. Wait, is let

Step-Level Speculation

Therefore, we have b*c = 468. Wait, let me

Therefore, we have b*c = 468.\n\n

Therefore, we have b*c = 468. Wait, let me double verify.

Therefore, we have b*c = 468.\n\n Wait, let me verify.\n\n

Therefore, we have b*c = 468.\n\nWait, let me verify.\n\nWe have:\n\n

Therefore, we have b*c = 468.\n\nWait, let me verify.\n\n1. a = 4sqrt(30), from AI.\n\n 1. AI = 2 sqrt(39). Then, AI\u00b2 = 156.\n\n

Figure 4: Illustration of hybrid approach.

