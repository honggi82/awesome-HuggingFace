# arXiv:2412.20993v2[cs.LG]27May2025

## Efficiently Scaling LLM Reasoning with Certaindex

Yichao Fu1∗ Junda Chen1∗ Siqi Zhu2 Zheyu Fu1 Zhongdongming Dai1 Yonghao Zhuang3 Yian Ma1 Aurick Qiao4 Tajana Rosing1 Ion Stoica5 Hao Zhang1 1UCSD 2Tsinghua University 3 Carnegie Mellon University 4Snowflake 5 UC Berkeley

### Abstract

Test-time reasoning algorithms such as chain-of-thought, self-consistency, and MCTS enhance LLM problem-solving but can wastefully generate many tokens without improving accuracy. At the same time, we observe that these algorithms exhibit answer stabilization: their intermediate solutions often cease to change after a certain point, and further investment of compute does not change their final answer. To quantify this phenomenon, we introduce Certaindex, an algorithm-agnostic metric measuring this evolving stability, signaling when further computation is unlikely to alter the final result. Certaindex is lightweight, can accelerate reasoning program inference via early exit, and further enables dynamic token allocation, gang scheduling, and many opportunities when integrated with real-world LLM serving systems. To quantify real-world benefits, we built Certaindex as a scheduler into Dynasor, our reasoning-aware LLM serving system, and demonstrate up to 50% compute savings and 3.3× higher throughput in real workloads with no accuracy drop. Our code is available at https://github.com/hao-ai-lab/ Dynasor.git.

### 1 Introduction

Large language models (LLMs) have recently demonstrated remarkable ability in solving complex problems. Central to these advances are test-time scaling algorithms [1; 2; 3], which allocate additional inference resources to progressively enhance model accuracy. Whether the reasoning algorithm is externally programmed (e.g. Self-Consistency [4], MCTS [5; 6], etc.) or internalized in the model (Chain-of-Thought based models such as Deepseek-R1 [7]), or a combination of both (as used in OpenAI-o3 [8], Grok3 [9], Gemini 2.5 Pro [10]), these methods fundamentally empower LLMs to tackle challenging problems more effectively.

However, we observe that LLM reasoning is highly token-inefficient, i.e., often leads to token overuse without further accuracy gains. This phenomenon is most observable in reasoning models: models like Deepseek-R1 can generate excessive amount of tokens, resulting in substantial 3x more tokens than it actually needs (see Figure 2). Similar token overuse behavior is also observed in recent studies [11; 12]. This inefficiency arises because existing reasoning algorithms lack mechanisms to detect diminishing returns in accuracy and terminate inference early, thereby wasting resources without gains in solution quality.

To address this inefficiency, we leverage a key observation about LLM reasoning: LLMs often signal when they’ve “settled” on an answer during reasoning. In particular, we study reasoning models with Chain-of-Thought (CoT, §2.1): we force the model to periodically output a result in the middle of its reasoning process, and examine how these intermediate answers evolve across successive steps. We find these intermediate answers frequently stabilize; that is, they rarely change after a certain number of reasoning steps, regardless of whether the answer is ultimately correct or not. This highly indicates that the model itself has reached a point of “high certainty” that further computation is unlikely to alter the final result, justifying safe termination. Conversely, significant variance in the

*Equal contribution.

Preprint. Under review.

[Figure 1]

Accuracy vs Token Usage

78% fewer tokens

AccuracyonMATH500(%)

80

60

40

Qwen2.5 7B Instruct Model

20

Qwen2.5 7B Reasoning Model

0

0 1 2 3 Average Tokens Used (k)

- Figure 1: The token efficiency curve for the traditional model is much steeper than reasoning model.

Figure 2: DeepSeek R1 performance on AMC23 (upper) and AIME24 (lower) at varying token budgets (scoring lowest to highest over 10 attempts). (Left) Standard reasoning with late answer outputs. (Right) Early answer extraction using Probe-In-The-Middle technique.

outputs suggests the model remains uncertain and is still exploring different solution paths, warranting continued reasoning. By monitoring these signals, we can decide if we can terminate LLM reasoning early, saving computational resources without token waste.

Taking the certainty as a signal, we further explore advanced algorithms such as Self-Consistency (SC), Monte Carlo Tree Search (MCTS), and Rebase [13] – all of which have seen empirical success in improving CoT-based reasoning in practice. While these algorithms have distinct mechanisms, we found they share a fundamental behavior: as more computational resources (e.g., tokens, reasoning steps) are invested, their answers tend to stabilize. This common observation indicated that a general measure of this evolving confidence is possible. To formalize this as a unified metric, we introduce certaindex (§3.1) – a universal metric designed to capture this developing certainty. Certaindex generalizes certainty into a normalized confidence score that serves as a proxy to measure reasoning progress. Regardless of what the reasoning algorithm is, high certaindex values indicate that the model’s answer is unlikely to change significantly with further computation.

Even better, we find that certaindex offers a narrorw interface to real-world LLM serving engines. Because certaindex is light-weight, it enables opportunities including early exit, dynamic token allocation across reasoning queries, and gang scheduling in multi-stage reasoning programs, all with almost no scheduling overhead (§3.2). To this end, we develop Dynasor – a reasoning-aware end-to-end serving system that leverage certaindex to optimize compute usage in both batch and online serving (§3.3). Dynasor employs certaindex to adaptively allocate token budgets to reduce the compute cost, and enables gang scheduling to increase request SLO attainment and throughput. Our evaluations on various datasets, LLMs, and reasoning algorithms show that in batch inference, it saves up to 50% compute to reach the same overall accuracy; and in online serving, it sustains up to

- 3.3× more queries or achieves 4.7× tighter latency SLOs at the same attainment rates. In summary, this paper makes the following contributions:

- 1. We identify significant token overuse in concurrent reasoning models and introduce probe-inthe-middle, a method that extracts intermediate answers during CoT to detect convergence and enable early stopping – validated both empirically and theoretically.
- 2. We develop Certaindex, a unified metric that quantifies reasoning progress across diverse strategies (CoT, SC, MCTS, REBASE), enabling adaptive compute allocation at test time.
- 3. We build Dynasor as a reasoning-aware serving system that leverages certaindex to dynamically allocate tokens and co-schedule multi-stage reasoning steps, achieving up to 50% compute savings in batch serving and 3.3× throughput gains in online serving.

### 2 Systematic Token Overuse – A Case Study in CoT

Recent work has shown that allocating more compute at inference time — so-called “test-time scaling” — consistently boosts performance on hard reasoning tasks [8; 14; 15]. Methods such as Chainof-Thought (CoT) [16], Best-of-N sampling [17; 18], Self-Consistency (SC) [4], and search-based

👤

|Deepseek-Distilled Qwen2.5-7B|
|---|

[Figure 2]

[Figure 3]

Evaluate (1+2i)6-3i

Qwen2.5-7B

<think> Okay, so I need to evaluate the expression (1 + 2i)6 - 3i. Hmm, let me think about how to approach this. I remember that when dealing with complex numbers, multiplication comes before addition and subtraction, so I should handle the multiplication first and then subtract 3i. First, let me rewrite the expression to make sure I understand it correctly: (1 + 2i) multiplied by 6, and then subtract 3i from the result. So, it's (1 + 2i) * 6 - 3i. Alright, let's tackle the multiplication part first. I know that when multiplying a complex number by a real number, I can distribute the multiplication across both the real and imaginary parts. So, for (1 + 2i) * 6, I can multiply 6 by each term inside the

To evaluate the expression (1+2i)6-3i, we will follow these steps:

- 1. Distribute the 6 to both terms inside the parentheses.
- 2. Simplify the resulting expression.
- 3. Subtract 3i from the simplified expression. Let's start with the distribution: (1+2i) 6 = 1 * 6 + 2i * 6 = 6 + 12i Now, we substitute this back into the original expression: 6 + 12i - 3i Next, we combine the like terms (the imaginary parts):6 + (12i - 3i) = 6 + 9i So, the final answer is: 6+9i

|parentheses separately. Let's do that step by step. Multiplying 6 by 1 gives me 6. Then, multiplying 6 by 2i gives me 12i. So, putting those together, (1 + 2i) * 6 equals 6 + 12i. Now, I have to subtract 3i from this result. So, the next step is to take 6 + 12i and subtract 3i from it. Let me write that down: 6 + 12i - 3i. Subtracting 3i from 12i should be straightforward. 12i minus 3i is 9i. So, now the expression simplifies to 6 + 9i. Wait a second, let me double-check that to make sure I didn't make a mistake. So, (1 + 2i) * 6 is indeed 6 + 12i because you distribute the 6. Then, subtracting 3i would mean subtracting 3 from the coefficient of i, so 12 - 3 is 9. So, 6 + 9i seems correct.\n\n Hmm, is there anything else I need to do here? The problem just asks to evaluate the expression, which I think I have done. Let me just make sure that I didn't misinterpret the original expression. It was (1 + 2i)6 - 3i, right? So, yes, that should be (1 + 2i) multiplied by 6, then subtract 3i. Alternatively, sometimes people might interpret 6(1 + 2i) - 3i, which is the same thing. So, either way, the result is 6 + 12i 3i, which is 6 + 9i. \n\nI don't think I need to do anything else because the expression is fully simplified now. So, 6 + 9i is the result. Wait, let me make sure about the arithmetic. 12i minus 3i is definitely 9i. Yeah, because 12 - 3 is 9. So, that part is correct. And 6 remains as it is because there's no other real term to combine with. So, 6 + 9i is the final answer.\n\nI guess another way to check is to substitute the values into a calculator or use another method to verify. But since I'm just doing it manually, I can verify each step again. Multiplying 6 by 1 gives 6, 6 by 2i gives 12i. Then subtracting 3i gives 6 + 9i. Yeah, that seems consistent. So, I think I'm confident that the evaluated expression is 6 + 9i. I don't see any mistakes in my reasoning, so I believe that's the correct answer.**Final Answer**The value of the expression is 6 + 9i. </think> To evaluate the expression (1 + 2i)6 - 3i, we start by distributing the multiplication across the complex number. First, we multiply 6 by each<br><br>|
|---|

Final Answer

First Answer

Double Checking

Completeness Check

Reassessment

Perspective Bridging

Re-verification

Confidence Building

term in the complex number 1 + 2i: (1 + 2i) * 6 = 6 * 1 + 6 * 2i = 6 + 12i. Next, we subtract 3i from the result: 6 + 12i - 3i Combining the imaginary parts, we get: 6 + (12i - 3i) = 6 + 9i. Thus, the value of the expression is

Confident

|6 + 9i|
|---|

- Figure 3: An Example of Self-Doubt Comparing a Reasoning Model (Deepseek-distilled Qwen-2.5

- 7B) vs. a Traditional Model (Qwen-2.5 7B) on a Problem from the MATH500 Dataset

algorithms (e.g., MCTS [5; 6], guided beam search [19], or REBASE [13]) all exploit more token usage to trade compute for accuracy.

While advanced reasoning algorithms boost accuracy on complex tasks, they often suffer from severe token inefficiency. To quantify this, we compare a Qwen2.5 7B (Non-Reasoning) [20] instruct model against a Qwen2.5 7B reasoning model [21] on the MATH-500 dataset, gradually increase the maximum token budget for each question, and track the accuracy change for the dataset. Figure 1 reveals that the reasoning model, despite eventually achieving higher peak accuracy, required up to

- 4.5× more tokens to reach the same accuracy as the instruct model.

Why do reasoning models overuse so many tokens? To answer this question, we conduct a case study focused on CoT reasoning models in math datasets (i.e., AMC23, AME24) to understand where this overuse occurs and why. We develop and use “Probe-In-The-Middle” (§2.1)—periodically inserting a prompt such as “Oh, I suddenly got the answer to the whole problem, Final Answer: boxed{"—to force the model to output its answer mid-reasoning, and record whether this answer is correct.

- Figure 2 presents the LLM’s accuracy for each question at various token budget allocations with (left) and without (right) “Probe-In-The-Middle”. Each row in this figure gives us an approximation of the number of tokens spent (left) vs actually needed (right) to arrive at the correct answer. For example, on AMC23, the model produces a median of 2.7K tokens, but can typically output the correct answer by a median of 830 tokens.

We further zoom in to one case, and realize the model the major source of inefficiency stem from what we called “self-doubt”. Figure 3 shows an annotated output where a reasoning model (Deepseekdistilled Qwen2.5-7B) answers a simple math question. Even as the reasoning model reached the correct answer very early (∼ 300 tokens), it continues the reasoning process, spending excessive tokens re-evaluating already correct answers, checking completeness, re-verifying assumptions, building confidence, etc. Several recent studies [11; 12] have also found similar phenomenon, indicating that this pattern of “self-doubt” is not an isolated issue specific to a single model or dataset but rather reflects a systematic tendency in current reasoning models.

#### 2.1 Addressing Token Overuse In Long CoT Reasoning by Probing-In-The-Middle

How do we address token overuse? We conjecture that LLMs, during their reasoning, will emit measurable signals to their internal reasoning state. Indeed, much prior work indicates that LLMs might possess an internal awareness of their answer confidence, akin to ‘LLMs knows when they know’ [22]. If these signals can be accurately detected, we can use them to stop the reasoning process without sacrificing accuracy, thus reducing token usage.

Our high-level idea is that if an LLM consistently produces the same intermediate answer over a sequence of reasoning steps, it indicates a high degree of certainty or stability for that answer, whether or not that answer itself is correct or not. Once such stability is reached, additional reasoning is unlikely to alter the outcome, presenting an opportunity for early termination. Conversely, if

[Figure 4]

- Figure 4: Illustration of Dynasor on CoT: (1) Probe-In-The-Middle for answer extraction, (2) early exit based on certainty (case 1), (3) post-generation validation for hesitation words (e.g., “wait”) (case 3), and (4) continue if not certain enough (case 2)

intermediate answers remain highly varied or fail to converge after significant effort, it may signal the problem as intractable for the model, also justifying early termination. Given this insight, we use “Probe-In-The-Middle” to first capture intermediate answers (e.g. every 64 tokens) as the model reasons, and then assess the certainty based on the consistency of these probed answers to decide whether to stop generation. By monitoring this certainty, we can truncate the reasoning chain, thus reclaiming wasted compute without sacrificing accuracy.

Probe-In-The-Middle. Figure 4 shows the high-level process. To capture whether the reasoning LLM has settled on its final answer, we interleave periodic “probes” into its reasoning process. Specifically, after a set number of generated tokens, we append the prompt: “Oh, I suddenly got the answer to the whole problem. Final Answer: boxed{", and record the output answer. This prompt forces the model to take a guess for the final answer at this specific point in its reasoning. Note that the exact phrasing of the extraction prompt is not critical. What matters is that it effectively guides the model to produce an answer immediately.

Concretely, we split the full reasoning chain into m steps, Y1,Y2,...,Ym, where each Yk corresponds to a fixed token interval (e.g., 64 tokens). At each step k, we sample Yk ∼ Pk Yk | x, Y1,...,Yk−1 . All probe tokens and their responses are then discarded before resuming the original decoding path. This lightweight probing mechanism allows us to pinpoint exactly when the model’s answer converges—enabling systematic comparison of token efficiency across models on large datasets.

Certainty Assessment via Answer Consistency. Using the answers captured through Probe-In-TheMiddle, we perform an immediate consistency check to assess the model’s certainty at each probing step. Concretely, let {y1,...,ym} be the sequence of probed answers from step 1 to step m. We measure consistency over a sliding window of width w steps: Ck = w1 kj=k−w+1 I yj = yk , where I[·] is the indicator function. Once Ck ≥ τ for some threshold τ ∈ (0,1], we deem the model sufficiently certain and terminate generation at step k (case 1 in Figure 4). Otherwise, we will continue reasoning (case 2 in Figure 4) until the budget is drained or some other stopping criteria is matched.

Post-Generation Validation. In addition to the answer consistency, we found that some linguistics markers like “wait” or “hmm” also indicates uncertainty. If we find these uncertainty indicators in the probed answers (case 3 in Figure 4), we will mark this answer as unconfident and omit this responses from the consistency test. This validation mechanism works synergistically with the consistency assessment to create a robust certainty metric.

Theoretical Grounding and Efficacy for CoT. The rationale for early terminating based on this observed stability is not merely empirical. In § 3.4, we provide a theoretical analysis. This analysis demonstrates that if a reasoning chain consistently yields the same answer across several consecutive reasoning steps, then as the number of these consistent steps increases, the observed answer provably converges to the final answer the CoT would have produced with full, unconstrained generation.

Applying “Probe-In-The-Middle” to CoT reasoning on datasets like MATH-500 (§4.1) demonstrates significant token efficiency gains. Identifying answer convergence allows early termination, which reduces computational overhead while maintaining improving accuracy. This success with CoT motivates certaindex, our universal metric extending the certainty-driven early exit concept to a broader spectrum of reasoning algorithms including Self-Consistency, MCTS, and Rebase.

### 3 Certaindex in General Reasoning Algorithms

#### 3.1 Generalizing Certainty to a Broad Spectrum of Reasoning Algorithms

Beyond just CoT, a wide spectrum of LLM reasoning algorithms also exhibit (or can be equipped with) measures that reflect their progress towards a stable answer. For instance, iterative refinement methods (e.g., CoT) may show convergence in output, while search-based algorithms (e.g., MCTS) might reach a state where further exploration yields diminishing returns. Recognizing this common signal of “certainty” or solution stability across different algorithms, we propose certaindex-—a unified, algorithm-agnostic confidence metric. High certaindex indicates close proximity to a solution or that additional computation is unlikely to improve the outcome.

While previous research has explored uncertainty estimation through various approaches (semantic metrics[23; 24], log-probability entropy [22; 25], and hidden state analysis [26; 27; 28]), certaindex offers a practical, integrative measure of model confidence suitable for direct implementation in LLM serving engines (e.g., SGLang and vllm). This facilitates a scheduling layer optimized for LLM reasoning workloads. In the following sections, we will define specific instantiations of certaindex for two common reasoning algorithm archetypes.

Certaindex in reasoning algorithms with multiple reasoning paths. Reasoning algorithms like Self-Consistency (SC), MCTS, and REBASE expand multiple reasoning paths to derive aggregated final answers (Appendix A). To quantify LLM’s certainty among these paths, we employ semantic entropy [23]. Given a question, n reasoning paths are generated and clustered into m groups based on their answers: C1,C2,...,Cm, where |Ci| denotes the number of paths in answer group Ci. The semantic entropy is calculated as H = − mi=1

|Ci|

n log |C

i|

n . We normalize H by its maximum log n to obtain certaindex: H˜ = loglogn−Hn ∈ [0,1]. For closed-form tasks (e.g., arithmetic or multiplechoice), we group outputs by exact string matching; for open-ended generation (e.g., code [29] or flexible mathematical expressions [30]), we compute pairwise similarities with a small embedding model [31] and cluster. Both approaches incur little to no overhead compared to LLM inference.

Certaindex in reasoning with a reward model. For reasoning algorithms that incorporate a reward model (e.g., MCTS, REBASE), we simply use the reward model’s normalized output R ∈ [0,1] as a measure of certainty. This approach builds on prior research demonstrating that reward signals can effectively guide resource allocation in program execution [32]. We collect the terminal reward scores from each reasoning path and aggregate them to compute certaindex: for MCTS, we take the average reward; for REBASE, the maximum reward. A higher aggregated reward indicates stronger certainty in the reasoning paths’ validity. These reward scores are obtained during normal execution and therefore incur no extra overhead during LLM inference.

#### 3.2 Effectiveness of Certaindex for Dynamic Token Budgeting in Reasoning Optimization

Since certaindex tracks the model’s self-assessed proximity to a final solution, it can effectively guide token budgeting to prevent unnecessary token expenditure. In this section, we empirically validate this core hypothesis: that higher certaindex values consistently predict fewer remaining token needs to reach a correct solution. This relationship holds across various models, reasoning algorithms, and datasets, forming the basis for smart token allocation.

Correlation between Certaindex and Remaining Computational Effort. Our hypothesis is that as a model becomes more “certain” (as measured by certaindex), it should be closer to outputting its final, stable answer. To test this, we compare these certaindex values against the “oracle” number of additional tokens the model actually required from that point to arrive at the correct final answer

(for solvable queries). This analysis spanned 4 different model–algorithm–task combinations, with representative examples of these correlations shown in Figure 5 (further examples are in Appendix B). Across all combinations, for solvable queries, we found Pearson correlation coefficients ranging from 0.17 to 0.75, with a strong mean of 0.52. This consistently positive correlation indicates that a higher certaindex value is indeed a reliable predictor of solution proximity, requiring fewer additional tokens.

Thresholding-based allocation. An immediate application of certaindex for token budgeting is a thresholding-based allocation policy. If a query reaches a high certaindex value early in its reasoning, it suggests the model is confident in an answer. We select a certaindex cutoff threshold at a chosen reasoning step (e.g., after a fixed number of reasoning steps, visualized as the horizontal orange line in Figure 5). Any query whose certaindex surpasses this threshold at this step is immediately halted. This approach has two benefits: (1) it efficiently reduces token usage on straightforward cases that quickly achieve high certainty, and (2) it halts unproductive paths early, preventing wasted resources on unsolvable or confidently incorrect answers. As our empirical results show, this approach often achieves little to no drop in accuracy on the broader set of solvable queries that do not cross this early exit threshold. Thresholding-based allocation is not strictly optimal, as we show in the pareto-frontier optimal strategy in Appendix B and Figure 12. Nevertheless, thresholding-based allocation is simple and efficient in practice, making it a compelling choice for real-world deployments.

[Figure 5]

[Figure 6]

- Figure 5: Correlations between certaindex strength (y-axis) and ground truth steps to solution (xaxis) with (algorithm, LLM) settings where algorithm ∈ {SC, Rebase, MCTS, CoT} with LLM ∈ {Llama [33] and QWQ [34]} on GSM8K. Certaindex is measured at the reasoning step marked by the red line. The orange line shows the thresholding-based allocation. The green line shows a more fine-grained approach through curve fitting. All plots except MCTS have both certaindex values and oracle steps averaged across multiple runs to combat randomness. See Figure 12 for the full result.

#### 3.3 Dynasor: A Certaindex-Driven Adaptive Compute Scheduler

The previous section demonstrated how Certaindex effectively quantifies certainty during LLM reasoning, enabling dynamic token allocation and reducing waste. Building on this, we introduce Dynasor – a reasoning-aware, end-to-end serving system that leverages Certaindex for adaptive scheduling. In real-world deployments for LLM reasoning, a single user query often spawns multiple related sub-queries, which we refer to as “reasoning programs”. These programs consist of logically connected requests that share context and computation. Traditional schedulers treat these sub-queries independently, missing opportunities for optimization. Dynasor addresses this by introducing a lightweight, Certaindex-driven scheduler that optimally manages both individual requests and entire reasoning programs. Beyond simple early exits, Dynasor implements two other key mechanisms to accelerate inference:

Certaindex-Driven Adaptive Compute Allocation. Dynasor dynamically adjusts token budgets for individual requests based on their real-time Certaindex values. As validated in §3.2, a high Certaindex indicates that the answer has stabilized, allowing the scheduler to reduce the token budget and save resources. This fine-grained control minimizes waste on queries that have converged or are unproductive, enhancing overall efficiency.

Gang Scheduling. Reasoning programs usually generate multiple related requests in parallel or at differet stages. Gang scheduling batches or prioritizes these requests together to maximize shared computation (e.g., KV-cache utilization) or early-exit high-certaindex programs to free resources. This mechanism reduces latency and improves throughput by aligning resource allocation with real-time reasoning progress.

Implementation. Dynasor is implemented as a lightweight scheduling component in SGLang [35]. For each decoding step, Dynasor monitor certaindex to reallocate resources or terminate requests, and use gang scheduling to prioritize requests within a reasoning program. Dynasor only introduces ∼500

lines of code into the core system, and requires no change to the model weights, reasoning algorithms, or the core execution backend. It acts as a thin layer around the standard decoding loop, yielding substantial gains in latency, throughput, and overall token efficiency. We describe the detailed design and implementation of Dynasor in Appendix D.

#### 3.4 Theoretical Basis for Early Exiting in CoT without Accuracy Loss

This section gives a sketch of the theoretical foundation for our early exiting strategy based on “Probe-In-The-Middle”. We show how it can terminate CoT reasoning without accuracy degradation, and we believe this theoretical grounding can be generalized to other reasoning algorithms. See Appendix E for a riguous proof.

Sketch of Proof. Our method assumes that the next-token distributions Pt(Yt+1|x,Y1..t) in a reasoning chain eventually converge to a stationary distribution P∗(Y |x). Once Pt = P∗, further computational steps are redundant. The “Probe-In-The-Middle” technique empirically detects this convergence. If observed answers (and thus their empirical distributions Pˆ) stabilize across several probing steps, it indicates the underlying true distributions Pt (and their mixtures P¯) have likely converged. To analyze this, we define mixture distributions:

Definition 1 (Mixture Distributions). Consider n samples, each corresponds to Pt,...,Pt+n. Denote the mixture distribution of samples i + 1,...,i + k ∈ [1,...,n] to be P¯ii+k = k1 kj=1 Pi+j.

Our formal justification shows that if our empirical stopping criterion is met, the empirical mixture distributions Pˆ are ϵ/3-close (in Total Variation distance) to the true mixture distributions P¯. This requires a sufficient number of probes k, as established by the following concentration bound:

- Lemma 1. If k = Ω M+log(1ϵ2 /δ) , then TV(P¯ll+t,Pˆll+t) ≤ ϵ/3, for all l = i + 1,...,i + k, and t = k − 1,k, with 1 − δ probability, assuming we have M disjoint groups of the outputs.

The empirical stopping criterion is triggered when the TV distance between consecutively estimated mixture distributions, Pˆii+k and Pˆii++jj+k (and similarly for mixtures of size k − 1), remains below a small threshold ϵ′ = ϵ/3 for all 1 ≤ j ≤ k (or 1 ≤ j ≤ k − 1 respectively). When this empirical stability is observed, and k is sufficiently large as per Lemma 1 (ensuring TV(P,¯ Pˆ) ≤ ϵ/3), the triangle inequality implies that the true underlying mixture distributions P¯ii+k and P¯ii++jj+k are also ϵ-close to each other (i.e., TV(P¯ii+k,P¯ii++jj+k) ≤ ϵ).

The connection between the stability of these true mixture distributions and the stability of individual distributions Pt (which ultimately implies convergence to P∗ given Assumption 1) is provided by:

- Lemma 2. If TV(P¯ii+k,P¯ii++jj+k) = 0, for any 1 ≤ j ≤ k, and if TV(P¯ii+k−1,P¯ii++jj+k−1) = 0, for any 1 ≤ j ≤ k − 1, then Eq. (2) holds for any t ≥ i, and T < 2k − 1.

Therefore, observing empirical stability with a sufficiently large k indicates that the true individual distributions Pt have stabilized and are ϵ-close to the stationary distribution P∗(Y |x). This justifies that early termination preserves accuracy up to a tolerance ϵ. The effective k must satisfy both the concentration requirement (Lemma 1) and be large enough relative to k∗ from Assumption 1.

### 4 Evaluation

To show the effectiveness of certaindex in real-world workloads, we evaluate certaindex and the endto-end serving system Dynasor in CoT [16] and three other reasoning algorithms (SC [4], REBASE [13], MCTS [6; 5]) on diverse datasets [36; 37; 38; 29; 20; 39]. For brevity, we provide detailed experimental setup in Appendix F.

Our experiments try to answer two main questions: (1) Batch workload: How effectively does certaindex reduce token consumption while preserving accuracy in real-world batch inference workloads? (§4.1); (2) Online workload: To what extent does Dynasor improve end-to-end performance in online serving scenarios? (§4.2). In addition, we perform ablation studies in (§4.3) to compare certaindex against other signals and selection of certaindex thresholds at runtime. We also perform a few more Dynasor system ablation studies in Appendix G.

#### 4.1 Batch Inference with Certaindex

To evaluate the effectiveness of certaindex in batch inference, we evaluate Dynasor on CoT with reasoning models, and compare the tokens-accuracy tradeoff curve for different setups. We also evaluate certaindex on other reasoning programs (SC, MCTS, REBASE) to see if the effect generalizes. See detailed experiment setup in Appendix F.1 for CoT, and Appendix F.2 for SC, MCTS and Rebase.

CoT. Figure 6 shows our approach achieve significant token savings across all configurations, reducing token usage by 11-29% while maintaining same accuracy as the baseline. For the 10% of problems where our method achieves the highest token reduction, we observe savings of 34% on AIME and 53% on MATH500. In particular, for the top 1% of problems, we achieve even more substantial reductions of 53% on AIME and 81% on MATH500. A similar result is shown in Deepseek-R1 (See Appendix G.1). The primary gains of the token savings comes from the effective probing mechanism to identify self-doubt in CoT reasoning, and using certaindex to early-stopping.

Baseline Ours (32) Ours (64) Ours (128) Ours (256) Ours (320)

###### 7B

###### 14B

32B

50

65

70

Accuracy(%)

| | | | |-11|%| |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | |-17%| |
|---|---|---|---|---|
| | | | | |
| | | | | |

AIME24

-15%

45

60

65

40

55

60

6 7 8 9 10

6 7 8 9 10

6 7 8 9

90

95

95

Accuracy(%)

| | | |-2|0%| |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | |-2|4%| |
|---|---|---|---|---|
| | | | | |
| | | | | |

###### -25%

AMC23

85

90

90

80

85

3.5 4.0 4.5 5.0 5.5 6.0

3.5 4.0 4.5 5.0 5.5

3.5 4.0 4.5 5.0 5.5

95

Accuracy(%)

| | | | | | |
|---|---|---|---|---|---|
| | |-2|9%| | |
| | | | | | |

| | | | |
|---|---|---|---|
| | |-18%| |
| | | | |

| | |-19%| |
|---|---|---|---|
| | | | |
| | | | |

90

Math500

90

90

80

85

85

1.5 2.0 2.5 3.0 3.5

2.0 2.5 3.0 3.5

2.0 2.5 3.0

Tokens (K)

Tokens (K)

Tokens (K)

- Figure 6: Comparing Dynasor Token-Accuracy Curve Across Deepseek Qwen-Distilled Model Scales (7B, 14B, 32B) and Datasets (AIME24, AMC23, Math500)

0.5 1.0 1.5 2.0 2.5

30

35

40

SC LiveCodeBench

2.0 2.5 3.0 3.5

- 72

- 73

- 74

MCTS ASDiv

0.5 1.0 1.5 2.0

45.0

47.5

50.0

Rebase Math

2.0 4.0 6.0 Number of Tokens (1e6)

90

92

SC GSM8K

4.0 6.0 8.0 Number of Tokens (1e5)

- 69

- 70

- 71

MCTS GSM8K

0.2 0.5 0.8 1.0 Number of Tokens (1e7)

- 87

- 88

Rebase GSM8K

Accuracy(%)

Ours Baseline-length Baseline-even

- Figure 7: Token-to-accuracy metric on batch processing workloads. Mean performance and std (error bars) of 10 runs are reported. Baseline-even allocates resource uniformally across all reasoning program. Baseline-length uses Detect@knob (Table 1) as the program’s process signal. SC, MCTS, Rebase. To show our method generalizes to other reasoning algorithms, we also compare Dynasor against a modified SGLang intra-program scheduler using the following policies (1) baselineeven, which allocates resources uniformly; and (2) baseline-length which uses Detect@knob, the cumulative tokens generated at a specific step (Table 3) as the proxy.

- Figure 7 show certaindex consistently reduces token usage by 9–52% across workloads compared to both baselines, without loss in accuracy. Notably, we achieve over 47% savings on SC-GSM8K and over 50% on REBASE-Math, highlighting Dynasor’s efficiency across diverse benchmarks. These gains primarily come from the intra-program scheduling algorithm(§ D.2.1), which accurately identifies high-certaindex programs and terminates them early without accuracy loss. This early termination strategy significantly reduces resource consumption by eliminating unnecessary sampling compared with baseline-even. In contrast, baseline-length leads to accuracy degradation even with less aggressive compute pruning, highlighting the effectiveness of certaindex-based resource allocation.

#### 4.2 Online Serving with Certaindex

We evaluate Dynasor against SGLang [35] and Parrot [40] with different inter-program schedulers. We use P90 deadline attainment as the SLO attainment (detailed setup in Appendix F.3). Each row in

- Figure 8 presents a key performance trade-offs in online serving: (a) Program arrival rate vs SLO attainment: what percentage of programs can meet the SLO as the arrival rate increases; (b) SLO scale vs SLO attainment: how tightly a service provider can configure the SLO while maintaining reliable attainment. (c) Tradeoff between Accuracy and SLO attainment.

Ours SGLang Parrot

Self-Consistency Math

MCTS ASDiv

Rebase GSM8K

100

100

100

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

| |
|---|

| |
|---|

80

| |
|---|

0

50

| |
|---|

4 6 8 Program Rate (program/s)

5 10 15 Program Rate (program/s)

0.5 1.0 1.5 2.0 Program Rate (program/s)

SLOAttainment(%)

100

100

100

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

| |
|---|

| |
|---|

0

0

0

2.5 2.0 1.5 1.0 0.5 SLO Scale

1.5 1.0 0.5 SLO Scale

10.0 7.5 5.0 2.5 SLO Scale

100

100

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | || |
|---|
<br><br>|| |
|---|
<br><br>| | |
| | | | | | | | |

| |
|---|

| |
|---|

| |
|---|

80

0

0

53 54 55 56 Accuracy (%)

72.5 73.0 73.5 74.0 Accuracy (%)

87.4 87.6 87.8 88.0 Accuracy (%)

- Figure 8: Evaluation on 3 online workloads on Dynasor against baselines. Rows from top to bottom: (a) Program rate vs SLO attainment, (b) SLO scale vs SLO attainment, (c) Accuracy vs SLO Attainment.

- (a) Rate vs. SLO attainment. Figure 8(a) shows that Dynasor achieves much higher sustainable request rates under P90 deadline attainment: 1.6 − 3.3× and 1.6 − 3.2× compared to SGLang and Parrot, resp. These gains stem from two key factors. First, our intra-program scheduler identifies and early-terminates programs with high confidence, freeing resources for pending requests. Second, our inter-program scheduler prioritizes requests from the same program, increasing turnover rate.
- (b) SLO scale vs. SLO attainment. Figure 8(b) shows deadline attainment under a fixed request rate for different systems. At these rates, Dynasor allows tighter SLO scales: 1.3 − 4.7× tighter than SGLang, and 1.7 − 3.3× tighter than Parrot. The source of gain is similar to (a): intra-program scheduler early-terminates requests and increase turnover rate, both increases effective request rate.
- (c) Accuracy vs SLO attainment. Figure 8 (c) (details in Appendix F.3) shows Dynasor achieving 0.7% - 2% higher accuracy than SGLang and Parrot across all three workloads at the same SLO attainment. This improvement results from Dynasor’s ability to redistribute compute between simple and hard queries, enabling it to solve more queries while maintaining SLOs that baselines could only match with significantly more compute.

Throughput. Our system shows equal average throughput (token per second) in all online settings compared to the baselines. The is because that under the given request rates, all workloads saturate the GPU memory and making the system memory bound. This also validates the introduced scheduler has no overhead.

2.0 4.0 6.0 Number of Tokens (1e6)

90.0

90.5

91.0

91.5

Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Self-Consistency GSM8K

Ours (Entropy=0.5)

Baseline-even

- Entropy=0.75

- Entropy=1

- Entropy=1.5

- Entropy=2

1.8 2.0 2.2 2.5 2.8 Number of Tokens (1e5)

72.5

73.0

73.5

74.0

| |
|---|

MCTS ASDiv

Ours (score=0.4)

Baseline-even

Score=0.05

Score=0.1 Score=0.7

- Figure 9: Performance comparison with different entropy threshold or reward score threshold.

Pearson Correlation

Kendall's Tau

0.8

0.68 0.69

0.61 0.59

0.6

0.40 0.40

0.35 0.34

0.4

LengthLogPEntropy(OurLinearComb.s)

LengthLogPEntropy(OurLinearComb.s)

Figure 10: Correlation between certainty measure and mean steps required on solvable problems.

#### 4.3 Ablation Studies

Choosing Different Thresholds. We show that certaindex threshold selection is vital. Fig. 9 demonstrates the impact of different certaindex thresholds. For SC on GSM8k, we select an entropy threshold of 0.5, as higher thresholds led to overly aggressive termination and accuracy degradation. Similarly, for MCTS on ASDiv, we choose a score threshold of 0.4, as lower thresholds (<0.4)

decreased accuracy while higher thresholds (>0.4) increased token consumption without proportional accuracy gains. These results highlight the importance of careful threshold selection in balancing compute efficiency and accuracy. In practice, Dynasor uses a profiler-guided approach to select these thresholds (see detailed discussion in Appendix D.2.2)

Choosing Different signals. One may ask if there exist other signals better than certaindex. We compare certaindex with two alternative metrics for estimating resource needs. (1) Reasoning path length: Longer reasoning paths (more tokens) often indicate harder problems, suggesting a potential correlation between path length and compute needs (e.g., more reasoning paths). (2) Mean normalized log probability: Established as a measure of LLM confidence [25], higher log probabilities may correlate fewer samples needed for correct answers in SC. Using the (SC, GSM8K, Llama3.1-

- 8B-instruct) setup, we evaluated certaindex alongside these metrics. Four metrics were tested: certaindex’s entropy measure H, mean output length [41], mean normalized log probability [25], and their linear combinations.

As shown in Fig. 10, H achieved the strongest correlation with ground truth compute requirements (Pearson Correlation of 0.68, Kendall’s Tau of 0.61), outperforming other metrics and matching complex combinations. These results confirm that certaindex is a simple yet effective proxy for estimating inference computational demands, offering robust performance across tasks and models. Our end-to-end token-to-accuracy evaluations in Appendix G compare different signals for resource allocation, confirming certaindex’s superior performance.

Other ablations. We also ablate the contribution to scheduling algorithm (§ G.2), fairness analysis (§ G.3), and comparing static thresholding vs fine-grained resource allocation (§ G.4).

### 5 Related Work

Prior works on LLM serving systems have advanced inference through system-level optimizations like batching [42; 43; 44; 45], memory paging [46], and PD disaggregation [47; 48], yet these typically treat requests as independent units. While more recent frameworks such as ParrotServe [40] and SGLang [35] address multi-request workflows by enabling dependency specification and KV-cache memory reuse, neither approach specifically targets LLM reasoning programs, which demand adaptive scheduling based on test-time scaling behaviors and statistical progress measures—capabilities uniquely provided by Dynasor. Concurrently, research into efficient LLM reasoning, particularly for long CoT processes suffering from excessive token generation, has explored techniques like modelmerging [49], mid-generation self-evaluation [50], specialized finetuning [11; 51; 52; 12; 53; 54], and question difficulty estimation [55], with various reasoning workloads [56; 20; 39]. However, while these methods enhance the accuracy-computation trade-off, their reliance on model modifications or additional external components often complicates production deployment. Distinctly, our approach employs a lightweight proxy variable, enabling a non-invasive and efficient scheduling layer that seamlessly integrates with existing infrastructure.

### 6 Limitation and Broader Societal Impact Discussion

Limitation. Our study primarily focused on optimizing token allocation through certaindex, but did not explore its integration with advanced serving techniques like PD disaggregation or chunked prefill. The potential impact of these integrations on the latency-accuracy trade-off in real-world workloads remains an open area for future research.

Broader Impact. Improvements of Dynasor in LLM inference efficiency can further democratize access to powerful models. By reducing computational costs and energy consumption without model changes, our method enables more sustainable and scalable LLM serving. However, we acknowledge potential risks, including bias in early exits, security vulnerabilities through side-channel attack for certaindex in multi-tenant settings, and manipulation risks if malicious inputs exploit the mechanism.

### 7 Conclusion

In this paper, we show that LLM reasoning algorithms frequently exhibit answer stabilization, leading to unnecessary token generation. We developed certaindex, an algorithm-agnostic metric, to detect this stability and enable early exit from the reasoning process. When we implement Certaindex as a scheduler into Dynasor, our reasoning-aware serving system, this simple yet effective mechanism results in significant practical benefits, including up to 50% compute savings and 3.3x higher throughput in production systems, without accuracy drop. Certaindex thus represents a key advancement in making sophisticated LLM reasoning more efficient and scalable.

### References

- [1] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [2] Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, and Zaid Harchaoui. From decoding to meta-generation: Inference-time algorithms for large language models. arXiv preprint arXiv:2406.16838, 2024.
- [3] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [4] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [5] Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023.
- [6] Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.
- [7] Deepspeed model implementations for inference (mii), 2023.
- [8] OpenAI. Openai o3-mini system card. January 2025.
- [9] xAI. Grok 3 beta — the age of reasoning agents, 2025.
- [10] Google DeepMind. Gemini 2.5 pro, 2025.
- [11] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024.
- [12] Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.
- [13] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models. arXiv preprint arXiv:2408.00724, 2024.
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [15] OpenAI. Learning to reason with llms, 2024.
- [16] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [17] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.
- [18] Robert Irvine, Douglas Boubert, Vyas Raina, Adian Liusie, Ziyi Zhu, Vineet Mudupalli, Aliaksei Korshuk, Zongyi Liu, Fritz Cremer, Valentin Assassi, et al. Rewarding chatbots for real-world engagement with millions of users. arXiv preprint arXiv:2303.06135, 2023.

- [19] Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. Self-evaluation guided beam search for reasoning. Advances in Neural Information Processing Systems, 36:41618–41650, 2023.
- [20] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [21] Deepseek. Deepseek-r1-lite-preview is now live: unleashing supercharged reasoning power, 2024.
- [22] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.
- [23] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664, 2023.
- [24] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.
- [25] Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. arXiv preprint arXiv:2002.07650, 2020.
- [26] Aviv Slobodkin, Omer Goldman, Avi Caciularu, Ido Dagan, and Shauli Ravfogel. The curious case of hallucinatory (un) answerability: Finding truths in the hidden states of over-confident large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3607–3625, 2023.
- [27] Hanyu Duan, Yi Yang, and Kar Yan Tam. Do llms know about hallucination? an empirical investigation of llm’s hidden states. arXiv preprint arXiv:2402.09733, 2024.
- [28] Gustaf Ahdritz, Tian Qin, Nikhil Vyas, Boaz Barak, and Benjamin L Edelman. Distinguishing the knowable from the unknowable with language models. arXiv preprint arXiv:2402.03563, 2024.
- [29] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [30] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.
- [31] N Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.
- [32] Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang, Peter Bartlett, and Andrea Zanette. Fast best-of-n decoding via speculative rejection. arXiv preprint arXiv:2410.20290, 2024.
- [33] AI Meta. Introducing llama 3.1: Our most capable models to date. Meta AI Blog, 2024.
- [34] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024.
- [35] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in Neural Information Processing Systems, 37:62557–62583, 2024.
- [36] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

- [37] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [38] Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772, 2021.
- [39] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.
- [40] Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, and Lili Qiu. Parrot: Efficient serving of {LLM-based} applications with semantic variable. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 929–945, 2024.
- [41] Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, and Hao Zhang. Efficient llm scheduling by learning to rank. arXiv preprint arXiv:2408.15792, 2024.
- [42] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for {Transformer-Based} generative models. In osdi, 2022.
- [43] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. arXiv preprint arXiv:2308.16369, 2023.
- [44] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, et al. Deepspeed-fastgen: High-throughput text generation for llms via mii and deepspeed-inference. arXiv preprint arXiv:2401.08671, 2024.
- [45] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al. {AlpaServe}: Statistical multiplexing with model parallelism for deep learning serving. In 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), pages 663–679, 2023.
- [46] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.
- [47] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. {DistServe}: Disaggregating prefill and decoding for goodput-optimized large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210, 2024.
- [48] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), pages 118–132. IEEE, 2024.
- [49] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [50] Rohin Manvi, Anikait Singh, and Stefano Ermon. Adaptive inference-time compute: Llms can predict if they can do better, even mid-generation. arXiv preprint arXiv:2410.02725, 2024.
- [51] Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067, 2025.

- [52] Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young Yun. Self-training elicits concise reasoning in large language models. arXiv preprint arXiv:2502.20122, 2025.
- [53] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. arXiv preprint arXiv:2501.12570, 2025.
- [54] Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Length-compressible chain-of-thought tuning. arXiv preprint arXiv:2502.09601, 2025.
- [55] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547, 2024.
- [56] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.
- [57] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.
- [58] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.
- [59] Kshiteej Mahajan, Arjun Balasubramanian, Arjun Singhvi, Shivaram Venkataraman, Aditya Akella, Amar Phanishayee, and Shuchi Chawla. Themis: Fair and efficient {GPU} cluster scheduling. In 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20), pages 289–304, 2020.
- [60] Skywork o1 Team. Skywork-o1 open series, November 2024.
- [61] Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. arXiv preprint arXiv:2305.05920, 2023.

### A Example of LLM Reasoning Algorithms

We describe four widely used reasoning algorithms. While their specifics differ, they share two core operations: (1) expansion: generating tokens to expand solution trajectories, and (2) aggregation: combining results from trajectories to derive a final answer. Increasing compute for expansion generally improves the quality of answers during aggregation.

MCTS

ICoT

Rebase

SC

At each level, search for # branch - # candidate nodes

P

P

P

P

I1 I2 I3

In

P Prompt A Answer

[Figure 7]

refine

Intermediate step

- 1. Selection

- 2. Expansion

- 3. Simulation

[Figure 8]

refine

Candidate Answer

T1 T2 T3

4. Back propogation

A

A

A

A

Knob - # path Knob - # branch

Knob - # refinement rounds

Knob - # search iterations

- Figure 11: Illustration of the workflow of different LLM reasoning algorithms discussed in Appendix A

Self-consistency (SC). Fig. 11(a) depicts the computational process of SC [4]. Starting with a prompt P, SC expands n trajectories T1,T2,...,Tn by sampling different outputs from the LLM (e.g., varying random seeds or temperature). Each trajectory represents a reasoning path that terminates in a candidate answer. During aggregation, SC applies majority voting across answers in T1:n and selects the one that appears most frequently. The key compute-control parameter is n, which dictates the number of generated trajectories.

Rebase. Rebase [13], as shown in Fig. 11(b), also begins by generating n independent outputs I1,I2,...,In from the input prompt P. Unlike SC, these outputs represent intermediate steps, which may or may not contain the final solution. They are ranked and assigned with scores s1,s2,...,sn, typically by a learned reward model. Rebase selects nodes with higher normalized exponential scores es

/ nj=1 es

j as parent nodes, from which the next n reasoning steps are branched out.

i

This process is repeated until n candidate solutions are returned. The final result is aggregated by (weighted) majority voting across all n candidate answers or selecting the top-scored answer (a.k.a. best-of-n). Rebase structures the reasoning process as a solution tree, where tree’s depth (i.e., solution steps to reach an answer) depends on LLM outputs and its width is exactly n, which controls its inference-time compute.

Monte Carlo Tree Search (MCTS). Starting from the prompt P, MCTS [6; 5] iteratively builds a solution tree (Fig. 11(c)). It expands nodes by sampling continuations from the LLM step-by-step, until reaching a leaf node containing a candidate solution. Each solution is scored via a reward model or simulation, and the score is back-propagated to its ancestral nodes to update their reward values. This completes one iteration (or rollout). MCTS iteratively performs n iterations, starting each from a node with the currently highest reward. n is the key knob that controls the inference-time compute – increasing n allows deeper and broader exploration of potential solutions.

Internalized Chain-of-thought (ICoT). Lastest LLMs, such as OpenAI o1 [15] and QwenQWQ [34], can internalize reasoning behaviors at training, without needing for explicit reasoning algorithms. These models generate extended chain-of-thoughts [16] sequences to derive candidate answers to reasoning queries. They iteratively refine these answers by reflecting on prior outputs and continuing sequential decoding until termination, yielding the final solution. The number of refinement rounds n, which corresponds to total decoded tokens, directly determines inference-time compute.

### B Certaindex Based Resource Allocation

Large language models (LLMs) have an inherent ability to “know when they know”, meaning they can self-assess their confidence in their answers. This ability, discovered in prior work such as [22], allows LLMs to indicate their certainty while generating tokens. In this section, We introduce

[Figure 9]

- Figure 12: Correlations between certaindex strength (y-axis) and ground truth steps to solution (xaxis) on 12 (algorithm, task dataset, LLM) settings where algorithm ∈ {SC, Rebase, MCTS, ICoT}, dataset ∈ {LiveCodeBench [29], GSM8K, ASDiv [38], GAME24 [30]}, and LLM ∈ {Llama [33], Gemma [57], Phi [58], QWQ [34]}. How certaindex is measured in each setting is shown in the y label of each plot. Certaindex is measured at the reasoning step marked by the red line. The orange line indicates the thresholding-based allocation. The green line illustrates a more fine-grained approach through curve fitting. For all plots (except MCTS), both certaindex values and oracle steps were averaged across multiple runs to combat randomness.

certaindex, a measure of this confidence, as a proxy for reasoning progress: high certainty suggests the LLM is nearing a final answer or directly indicates a lower absolute compute needed to reach a final answer, whether correct or not. Our goal is to measure and track certaindex during inference, enabling dynamic resource allocation, prioritization of complex queries, scaling back simpler queries, and terminating unpromising queries. Some of the contents will overlap with § 3.1, as we want to provide a more complete narrative of the certaindex definition.

Various methods have been proposed to estimate LLM uncertainty, including semantic measures [23; 24], log probability entropy [22; 25], and hidden state-based indicators [26; 27; 28]. While certaindex’s mathematical formulation varies across reasoning algorithms, its core interpretation remains the same: the LLM’s certainty in its reasoning paths.

#### B.1 Measuring Certaindex

This section presents two basic formulations of certaindex.

Certaindex in typical reasoning algorithms. Reasoning algorithms like SC, MCTS, and Rebase expand multiple reasoning paths to derive aggregated final answers (Appendix A). To quantify LLM’s certainty among these paths, we employ semantic entropy [23], which is derived from empirical entroy. Given a question P, n reasoning paths are generated and clustered into m groups based on their answers: C1,C2,...,Cm, where |Ci| denotes the number of paths in answer group Ci. The empirical entropy is calculated as: H = − mi=1

|Ci|

n log |C

i| n .

H intuitively measures the diversity of answer distributions. A higher H indicates greater uncertainty, where reasoning paths diverge into many different answers. Conversely, a lower H suggests higher certainty, typically when most paths converge to the same answer (large |Ci| for some group i). The maximum entropy max(H) = log n occurs when each path yields a unique answer (m = n,|Ci| = 1 for all i). We normalize H by max(H) to obtain certaindex:

max(H) − H max(H) ∈ [0,1]. (1)

H˜ =

For SC/Rebase/MCTS on exact-answer tasks such as arithmetic and multiple-choice, those exact answers can be extracted by applying string-matching on the generated outputs, and grouped based on their equality. In more open-ended generation tasks such as code (e.g., LiveCodeBench [29]) or

flexible mathematical expressions [30], answer extraction becomes non-trivial. We employ small embedding models [31] (e.g., 100M parameters) to compute textual similarities between final outputs, and cluster reasoning paths based on semantic proximity, which enables efficient answer grouping across varying problem formats.

Both clustering methods are computationally lightweight: string matching is fast and efficient; running the embedding model and calculating similarity remains computationally insignificant compared to LLM prefill and decode operations.

Certaindex in reasoning algorithms with a reward system. For reasoning algorithms that incorporate a reward model (e.g., MCTS, Rebase), we simply use the reward model’s normalized output R ∈ [0,1] as a measure of certainty. This approach builds on prior research demonstrating that reward signals can effectively guide resource allocation in program execution [32]. We collect the terminal reward scores from each reasoning path and aggregate them to compute certaindex. The aggregation method varies by algorithm: for MCTS, we use the mean reward across its different paths, while for Rebase, we take the maximum reward value. A higher aggregated reward indicates stronger certainty in the reasoning paths’ validity, while a lower score suggests uncertainty. These reward scores are collected during the normal execution of the reasoning algorithms and thus do not incur additional overhead.

Combining multiple certaindex indicators. Certaindex can also be composed by multiple signals, provided they effectively capture certainty during the LLM reasoning process. When multiple signals are available, they can be combined by applying individual thresholds to each metric. For instance, in our MCTS/GSM8K experiments (Fig. 12), we use two distinct metrics to measure certaindex: the reward score R and the entropy-based measurement H˜. Each metric is compared against its respective threshold, Rτ and H˜τ respectively. A program is considered to meet the certainty requirement only if all metrics exceed their thresholds.

#### B.2 Effectiveness of Certaindex

This section empirically demonstrates that certaindex correlates strongly with the computational resources required to reach correct solutions across diverse models, reasoning algorithms, and task datasets. Higher certaindex values consistently indicate lower total compute needs.

Correlation. We measure certaindex at intermediate reasoning steps and analyze its correlation with the ground-truth number of steps required to yield correct answers, as determined by an oracle. Queries are categorized as solvable or unsolvable, where unsolvable queries cannot produce correct answers even with near-infinite compute. Figure 12 illustrates this correlation across 12 (model, algorithm, task) settings. On solvable queries, Pearson Correlation values between certaindex and required compute range from 0.17 to 0.75 (mean 0.52), indicating a strong correlation between high certaindex and fewer steps needed to solve a query. We next demonstrate two potential resource allocation strategies using certaindex.

Thresholding-based allocation. We measure certaindex at a specific reasoning step, shown as a red vertical line in each plot of Fig. 12. A straightforward allocation strategy sets a threshold value t (orange horizontal lines) and halts inference for any query with certaindex exceeding t. In all plots, nearly no solvable queries fall in the upper-right area beyond the red and orange lines, indicating that an appropriately chosen t can effectively early-terminate queries once their certaindex is greater than t. For these queries, this reduces resource usage for solvable queries and prevents over-allocation to unsolvable ones (red dots above the orange line), without sacrificing accuracy.

Pareto-frontier allocation. While thresholding offers a coarse-grained control, a more nuanced approach is a Pareto-frontier allocation. Instead of a binary stop/go decision based on a single early checkpoint, this policy aims to assign a dynamically tailored computational budget to each query based on its evolving certaindex throughout the reasoning process. The underlying principle is to leverage the observed correlation to inform a more continuous resource allocation. Rather than simply “fitting a curve”, we establish an empirically-derived relationship that maps different certaindex values to an estimated optimal remaining token budget for that level of certainty. This mapping (conceptually represented by the green curve in Figure 12) reflects a trade-off: for higher certaindex values, the optimal additional budget is small, while for lower certaindex values, more computation might be warranted, up to a certain point. This allows for a dynamic allocation scheme where the maximum additional tokens for a query are continuously adjusted or capped based on its current certaindex.

Queries exhibiting high certaindex (signaling proximity to a stable solution) are allocated fewer future tokens, while those still showing lower certaindex (indicating ongoing exploration or difficulty) might be allowed more, guided by this learned relationship. This strategy seeks to optimize resource use more globally across all queries, distributing the computational budget more effectively, as further elaborated in our “smart scheduler” design (§G.4).

#### B.3 Comparing Certaindex with Other Signals

One may ask if there exist other signals better than certaindex. We compare certaindex with two alternative metrics for estimating resource needs. (1) Reasoning path length: Longer reasoning paths (more tokens) often indicate harder problems, suggesting a potential correlation between path length and compute needs (e.g., more reasoning paths). (2) Mean normalized log probability: Established as a measure of LLM confidence [25], higher log probabilities may correlate fewer samples needed for correct answers in SC.

Using the (SC, GSM8K, Llama3.1-8B-instruct) setup, we evaluated certaindex alongside these metrics. Four metrics were tested: certaindex’s entropy measure H, mean output length [41], mean normalized log probability [25], and their linear combinations. As shown in Fig. 13, H achieved the strongest correlation with ground truth compute requirements (Pearson Correlation of 0.68, Kendall’s Tau of 0.61), outperforming other metrics and matching complex combinations. These results confirm that certaindex is a simple yet effective proxy for estimating inference computational demands, offering robust performance across tasks and models. Our end-to-end token-to-accuracy evaluations in Appendix G compare different signals for resource allocation, confirming certaindex’s superior performance.

Pearson Correlation

Kendall's Tau

0.8

0.68 0.69

0.61 0.59

0.6

0.40 0.40

0.35 0.34

0.4

LengthLogPEntropy(OurLinearComb.s)

LengthLogPEntropy(OurLinearComb.s)

- Figure 13: Correlation between certainty measurements and mean steps required to solve problems on solvable problems. We obtain the ground-truth mean steps by solving the queries using the LLM multiple times and counting the average steps.

### C Characterizing the Relationship Between Certaindex and Detection Steps

- Fig. 14 demonstrates the relationship between certaindex values and program steps across different detection points in a single run, using SC on the GSM8K dataset with Llama3.1 8B Instruct model. The figure consists of six subplots, each representing a different detection step ranging from 5 to 30, indicated by “Detect @knob” values.

Each point in the plots represents an individual problem, categorized into three types: Early Stoppable Problems (pink), Solvable Problems (blue), and Unsolvable Problems (gray). The x-axis shows the number of steps taken, while the y-axis displays the certaindex value (C = H˜). A horizontal orange dashed line indicates the certaindex threshold value, and a vertical red dashed line marks the step at which certaindex is collected.

The consistent pattern across all detection points demonstrates a strong correlation between certaindex and required reasoning steps, with Pearson correlation coefficients exceeding 0.5 for solvable problems. As detection steps increase from 5 to 30, we observe a 30% increase in early-stopped problems, indicating that higher certaindex values reliably predict when the LLM is converging toward a final answer. Certaindex works as a reliable proxy for reasoning progress. However, this improved accuracy trades off against potential compute savings, as later detection points leave less opportunity for early termination. Despite this tradeoff, certaindex proves to be a reliable predictor of required reasoning steps across all detection timings, maintaining its effectiveness regardless of measurement timing.

Detect @knob = 5

Detect @knob = 10

Detect @knob = 15

Detect @knob = 20

Detect @knob = 25

Detect @knob = 30

1.0

1.0

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

0.8

0.8

0.6

0.6

0.6

0.6

0.6

0.6

C=

0.4

0.4

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0.0

0.0

0 5 10 15 20 25 Inf

0 5 10 15 20 25 Inf

0 5 10 15 20 25 Inf

0 5 10 15 20 25 Inf

0 5 10 15 20 25 Inf

0 5 10 15 20 25 Inf

Early Stoppable Problems Solvable Problems Unsolvable Problems Threshold Step To Collect Certaindex

- Figure 14: Certaindex Values Across Different Detection Steps in Self-Consistency Reasoning

D Dynasor: System Design

Dynasor is straightforward to use. Developers define reasoning programs through the provided abstraction, implementing certaindex and scaling knob control functions. These programs are submitted to the application runtime, which monitors certaindex values, dynamically adjusts resource allocation, and forwards requests to the system runtime for execution. Programs either scale up for further computation or terminate early when resources are no longer allocated.

Figure 15a illustrates Dynasor’s three-component architecture: (a) a Reasoning Program Abstraction (Program) offering a standardized interface for diverse reasoning algorithms, (b) an Application Runtime that dynamically allocates computational resources based on CertaIndex measurements, and (c) a System Runtime managing request-level scheduling on the backend infrastructure.

This architecture enables Dynasor to adapt computation dynamically to each query’s difficulty level, allocating resources proportionally to reasoning complexity rather than using static, one-size-fits-all token budgets.

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 15: Left(a): Dynasor Architecture. Middle(b): Reasoning Program Interface. Right(c): Example Program (SC).

- D.1 Reasoning Program Abstraction

A reasoning program maintains three runtime properties: (1) certaindex, the certainty measure of the reasoning progress; (2) knob, the intrinsic scaling factor of the program; and (3) state, which stores the intermediate variables and results in previous steps.

The Reasoning Program Abstraction (or Program) provides a unified framework for developers to define a variety of reasoning algorithms. It introduces a narrow interface for the program to interact with the application runtime. Fig. 15a shows the structure of a reasoning program. Developers only implement (1) update_certaindex(), which calculates and updates the certaindex at a particular inference step, and (2) execute(), which runs the reasoning algorithm (Fig. 15b).

Fig. 15c illustrate a simple example implementation of SC: the update_certaindex() function computes the entropy across different branches, and the execute() function iteratively expands the generation. After each iteration, it aggregates the result and update certaindex. This process iterates until the program depletes its resources or exits.

- D.2 Application Runtime

- D.2.1 Certaindex-Based Intra-Program Scheduler

The certaindex-based intra-program scheduler controls the resource allocation of the program at runtime. The scheduler lifetime is described as follows:

[Figure 13]

Figure 16: Illustration of Gang Scheduling

Initialization. When a new program arrives, the scheduler initializes it with a predefined maximum resource cap and allocates resources for its first run. It also establishes a resource scheduling policy to guide future allocations.

Resource Allocation. The scheduler continuously monitors each program’s certaindex and uses it to determine resource allocation. As programs run, they repeatedly request resources from the scheduler while updating their certaindex to reflect the progress of reasoning. When multiple programs run concurrently, the scheduler can prioritize resource allocation between them via certaindex-based intra-program allocation policy.

Termination. When a program’s certaindex exceeds a limit based on its allocation policy or reaches the maximum resource cap, the scheduler denies further resources allocation for this program. The program then receives a termination signal and aggregates results based on its generation history.

We use the SC in Fig. 15c to illustrate the scheduler behavior. The program is first submitted to the application runtime, where the intra-program scheduler is initialized and assigned the initial resource (5 branches) to the program. SC sets a simple certaindex threshold to determine whether the program should terminate. As the program expand, it updates certaindex, and request resource from the scheduler for next iteration. The scheduler checks the certaindex against the policy, and decide if it should allocate resource for the program to continue running.

Resource allocation policy. A resource allocation policy determines compute allocation (e.g., branches in SC and iterations in MCTS) for each program based on its certaindex. §B.2 discussed two certaindex-based allocation policies: simple thresholding and pareto-frontier, both implemented in the system. Additional alternatives are analyzed in §G.4. Developers can easily configure the intra-program scheduler to use other certaindex-based policies.

#### D.2.2 Profiler-Guided Policy Calibration

We explain how to determine (or calibrate) a resource allocation policy, focusing on a certaindex threshold policy as our example. Determining an effective resource allocation policy is critical for reasoning program performance (§G.4). Program submitters often lack insight into runtime patterns, and program characteristics may shift due to algorithmic changes or data distribution shifts.

Dynasor provides an optional profile-based hyperparameter tuner to help users identify optimal scheduling policies for certaindex-based resource allocation. Users submit a batch of programs with labeled data - such as verified answers to questions (e.g., math problem solutions), response rankings, or reward model scores. The profiler collects runtime metrics, including certaindex and token usage, to determine optimal resource allocation across different certaindex ranges while maintaining accuracy requirements. For threshold-based allocation, the profiler stops allocating resources when a program’s certaindex exceeds the threshold, and otherwise sets resources to a maximum cap. The threshold is calibrated to meet the accuracy requirements (e.g., not hurt accuracy in all calibration data points). This calibration process can be periodically executed in real-world serving scenarios to meet the dynamics of data distribution shifts.

#### D.3 System Runtime

- D.3.1 Program-Aware Inter-Program Scheduler

The Program-Aware Inter-Program Scheduler optimizes scheduling and memory management at the program level, reducing per-program latency through two key strategies: (1) gang scheduling algorithm to prioritize requests originated from the same program, and (2) approximate Shortest-JobFirst (SJF) scheduling algorithm to reduce head-of-the-line (HoL) blocking and improve per-request latency.

Gang Scheduling. Gang scheduling groups requests from the same program together to minimize stragglers and reduce overall completion time. Fig. 16 demonstrates the advantage of Gang scheduling over sequential scheduling using an example with two programs arriving at t=0, where the system has a batch size of 2. Each program has two requests: program 1’s requests take 4 ms each, and program 2’s take 5 ms each. By prioritizing one program at a time, Gang scheduling (left) reduces average latency from 9 ms to 6.5 ms compared to sequential scheduling (right).

Approximating Shortest Job First (SJF). Our inter-program scheduler implements an approximating SJF scheduling algorithm to mitigate HoL blocking and improve average per-program latency. A program’s total execution time depends on two key factors: total compute requirements (knobs, e.g., number of branches/iterations) and token length per branch/iteration. While per branch/iteration’s exact LLM generation lengths cannot be known in advance [41], we can estimate token length per iteration by leveraging program locality and using historical averages from previous iterations. This estimation, combined with the compute requirements, helps predict total execution time. In Dynasor, certaindex controls the total compute requirements (deciding how many iterations to run), while SJF uses the estimated token length per iteration to optimize execution order.

Starvation Prevention. Dynasor promotes finish-time fairness [59], which compares a program’s completion time in a shared system (Tshared) to its estimated independent completion time (Tindependent). This metric naturally suits LLM serving as it accounts for generation length. We define finish-time fairness as ϕ = Tshared/#output tokens in our case, which is approximately proportional to the standard finish-time fairness since Tindependent can be estimated as k · #output tokens when decoding dominates the generation time, where k is a constant. While Gang scheduling alone significantly improves performance without causing starvation, adding SJF optimization can potentially lead to starvation of programs with longer estimated lengths. Gang scheduling alone already provides substantial performance benefits, making it a viable standalone option. For deployments using SJF, we implement a priority escalation mechanism where programs that have been waiting too long receive elevated priority, effectively preventing starvation. As demonstrated in §G.3, Dynasor promotes fairness compared to other baseline scheduling strategies.

- D.3.2 Other System Runtime Components

Prefix Cache Manager. Dynasor leverages prefix cache sharing. Independent branches of a program share a unified prompt KV-cache, while dependent reasoning chains reuse their longest common prefixes. For algorithms with reward models, Dynasor reuses prefixes during reward evaluation, reducing latency. The manager also handles automatic cache eviction. When memory is constrained, the KV cache of the programs with no active requests is assigned lower priority and evicted first.

Program Context Manager is a thin wrapper that tracks registered programs and their runtime characteristics. It provides cache eviction hints to the prefix cache manager based on program behaviors. Unlike the static program DAGs used in SGLang/ParrotServe, this component offers more flexibility by supporting dynamic generation patterns required by algorithms like MCTS and Rebase.

Execution Backend manages the execution of LLM requests, optimizing model performance through techniques like CUDAGraph. This layer is designed to be adaptable to various execution engines including vLLM, TensorRT, and SGLang.

#### D.4 System Implementation

We build Dynasor on top of SGLang (version 0.3.3 post1). The intra-program scheduler is implemented as a Python library on the client side, while the inter-program scheduler is integrated into the server-side scheduler.

Using Dynasor, we adapt various reasoning algorithms (Appendix A) to the Program interface, including their custom certaindex implementations. Our system comprises approximately 1.5k lines of Python code, covering the Program interface, application runtime, and system runtime. Notably, the core system runtime only comprises around ∼ 500 lines of code change, and the changes are modular and non-invasive, making it a very clean implementation into the core serving system logic.

Adapting each reasoning program requires an additional 40 to 150 lines of code to define certaindex logic and integrate with the Program interface. This implementation overhead is minimal compared to the original implementations of these algorithms, which spans up to 4,000 lines of code.

### E Proof of Stopping Criteria for CoT and Probing

Given the input prompt x, we can stop the reasoning chain Y1,...,Yt,Yt+1 at time t if the following holds for any T > t.

Pt(Yt+1|x,Y1,...,Yt) =t Pt+1(Yt+2|x,Y1,...,Yt+1) t+1= ... =T PT(YT+1|x,Y1,...,YT) = P∗(Y |x).

(2)

The above definition means that adding new steps in the reasoning chain does not change the distribution of the generated answer anymore. As a result, the reasoning chain reaches the stationary distribution, P∗(Y |x), which is the desired output given the prompt x in the first place.

We assume that the language model will eventually converge in the chain of thought, if the answer distribution remains the same for long enough.

Formally it is expressed as follows. Assumption 1. If equality Pt(Yt+1|x,Y1,...,Yt) = Pt+i(Yt+i+1|x,Y1,...,Yt+i) holds for any 1 ≤ i ≤ k∗,

then Pt(Yt+1|x,Y1,...,Yt) = PT(YT+1|x,Y1,...,YT) = P∗ holds for any T > t. Further assume that k∗ = O(M), where M is the number of distinct output groups.

Based on the above assumption, we propose the test criterion in Definition 2.

- Definition 1. Consider n samples, each corresponds to Pt,...,Pt+n. Denote the mixture distribution of samples i + 1,...,i + k ∈ [1,...,n] to be P¯ii+k = k1 kj=1 Pi+j.

Let the smallest t where Eq. (2) holds be t∗. Then for any i,j ≥ t∗, P¯ii+k = P¯jj+k = P∗. On the other hand, if i < t∗, then there exists j > i, so that P¯ii+k ̸= P¯jj+k. This intuition is formally expressed in Lemma 2 below. We therefore propose to test the difference among P¯ii+k, P¯ii+1+k+1,...,P¯ii++2kk.

Lemma 2. If TV(P¯ii+k,P¯ii++jj+k) = 0, for any 1 ≤ j ≤ k, and if TV(P¯ii+k−1,P¯ii++jj+k−1) = 0, for any 1 ≤ j ≤ k − 1, then Eq. (2) holds for any t ≥ i, and T < 2k − 1. Proof. We first have from the definition of the TV distance that

P ¯ii+k − P¯ii++jj+k

= 0. For a general 1 ≤ j ≤ k, we have from the definition of P¯ that

1

j

1 k

= P ¯ii+k − P¯ii++jj+k

(Pi+l − Pi+l+k)

l=1

1

= 0.

1

Therefore, jl=1 Pi+l = jl=1 Pi+l+k. Take j = 1, then

Pi+1 = Pi+1+k. Plugging into the case where j = 2, then we have Pi+2 = Pi+2+k. Expanding the recursion for all 1 ≤ j ≤ k, we obtain that

Pi+j = Pi+j+k, ∀1 ≤ j ≤ k. (3)

Similarly from the assumption that TV P ¯ii+k−1,P¯ii++jj+k−1 = 0, for any 1 ≤ j ≤ k − 1, we have that:

0 = P ¯ii+k−1 − P¯ii++jj+k−1

and consequently that

1 k − 1

=

1

j

(Pi+l − Pi+l+k−1)

l=1

, ∀1 ≤ j ≤ k − 1,

1

##### Pi+j = Pi+j+k−1, ∀1 ≤ j ≤ k − 1. (4)

The equalities in Eqs. (3) and (4) form a graph over the vertices that are Pi+1,...,Pi+2k−1, where the equalities are the edges. Note that all the nodes are edge connected by virtue of Eqs. (3) and (4). Therefore, we have that

##### Pj = Pl, ∀i + 1 ≤ j,l ≤ 2k − 1.

| |
|---|

We therefore define the approximate stopping criteria as follows.

- Definition 2. We define the ϵ-accuracy of the stopping criteria:

TV(P¯ii+k,P¯ii++jj+k) ≤ ϵ, for any 1 ≤ j ≤ k; TV(P¯ii+k−1,P¯ii++jj+k−1) ≤ ϵ, for any 1 ≤ j ≤ k − 1.

We then ask: how many data samples k do we need, to be able to test our stopping criteria up to ϵ-accuracy? More concretely, we construct the estimator of P¯ll+t, for l = i + 1,...,i + k, and t = k − 1,k as follows. Assume we have M disjoint groups of the outputs. Denote each group of answer as Cm, for m = 1,...,M. Then we estimate the average probability mass function P¯ll+t by defining the Bernoulli random variable for each Cm: ZC

τ m = {Yτ ∈ Cm}, and

Pˆll+t(S) = 1t τ l+=tl+1 ZτS. The next lemma establishes that with k = Ω M+log(1ϵ2 /δ) , then TV(P¯ll+t,Pˆll+t) ≤ ϵ/3, for all l = i + 1,...,i + k, and t = k − 1,k, with 1 − δ probability. Then we just have to test that

TV(Pˆii+k,Pˆii++jj+k) ≤ ϵ/3, for any 1 ≤ j ≤ k; TV(Pˆii+k−1,Pˆii++jj+k−1) ≤ ϵ/3, for any 1 ≤ j ≤ k − 1.

If the above inequalities stand with the established number of steps k, then by the triangle inequality of the total variation (TV) distance, we obtain the ϵ-accuracy in the stopping criteria in Definition 2:

TV(P¯ii+k,P¯ii++jj+k) ≤ TV(P¯ii+k,Pˆii+k) + TV(Pˆii+k,Pˆii++jj+k) + TV(Pˆii++jj+k,P¯ii++jj+k) ≤ ϵ.

Lemma 1. If k = Ω M+log(1ϵ2 /δ) , then TV(P¯ll+t,Pˆll+t) ≤ ϵ/3, for all l = i + 1,...,i + k, and t = k − 1,k, with 1 − δ probability.

Proof. To prove convergence in the total variation (TV) distance, we first express it as follows. Let S belong to the power set (denoted as expC) of Mm=1 Cm. Then

TV(P,¯ Pˆ) = sup

P ¯(S) − Pˆ(S) .

S∈expC

We then prove convergence for a fixed set S, and then apply union bound to expC to obtain the final result.

Denote a Bernoulli random variable ZτS = {Yτ ∈ S}. Then Pˆll+t(S) = 1t lτ+=tl+1 ZτS. On the other hand, P¯ll+t(S) = 1t lτ+=tl+1 Pτ(Yτ ∈ S|x,Y1,...,Yτ−1), Note that Yτ|x,Y1,...,Yτ−1 ∼ Pτ(·|x,Y1,...,Yτ−1). Therefore,

E ZτS|x,Y1,...,Yτ−1 = Pτ(Yτ ∈ S|x,Y1,...,Yτ−1).

Table 1: Offline workload Configurations. LiveCodeBench (LCB), Llama2 7B [5], Skywork7B [60], Llemma 7B and 34B [13] are fine-tuned models used in different settings as LLM/reward model.

(Algo., Dataset, LLM) Reward Model # Samples Resource Cap (SC, LCB, Llama3.1 8B) / 400 5,10,15,20,25,30 (SC, GSM8K, Llama3.1 8B) / 1000 5,10,15,20,25,30 (MCTS, ASDiv, Llama2 7B) Skywork 7B 300 3,7,10,15,20 (MCTS, GSM8K, Llama2 7B) Skywork 7B 300 3,7,10,15,20 (Rebase, MATH, Llemma 7B) Llemma 34b 500 16,32,64,128

(Rebase, GSM8K, Llemma 34B) Llemma 34b 500 16,32,64,128

Table 2: Online workload Configurations

Algorithm Dataset LLM Reward Model Base Deadline (s)

SC MATH Llama3.1 8B / 240 MCTS ASDiv Llama2 7B Skywork 7B 60 Rebase GSM8K Llemma 34B Llemma 34b 300

This means that ZτS − Pτ(Yτ ∈ S|x,Y1,...,Yτ−1) is a martingale difference sequence. We also know that ZτS − Pτ(Yτ ∈ S|x,Y1,...,Yτ−1) ≤ 1. Therefore, by the Azuma-Hoeffding inequality, we have:

l+t

ϵ2 2t

ZτS − Pτ(Yτ ∈ S|x,Y1,...,Yτ−1) ≥ ϵ ≤ 2exp −

P

,

τ=l+1

or

l+t

l+t

1 t

1 t

P P ¯ll+t(S) − Pˆll+t(S) ≥ ϵ = P

ZτS −

Pτ(Yτ ∈ S|x,Y1,...,Yτ−1) ≥ ϵ

τ=l+1

τ=l+1

tϵ2 2

≤ 2exp −

.

Taking a union bound over S ∈ expC, we have:

P ¯ll+t(S) − Pˆll+t(S) ≥ ϵ

P TV(P¯ll+t,Pˆll+t) ≥ ϵ = P sup

S∈expC

tϵ2 2

≤ 2MP P ¯ll+t(S) − Pˆll+t(S) ≥ ϵ ≤ 2M+1 exp −

.

On top of that, taking a union bound over l = i + 1,...,i + k, and t = k − 1,k, we have that the probability of TV(P¯ll+t,Pˆll+t) ≥ ϵ for one of the combination of l = i + 1,...,i + k, and t = k − 1,k is less than:

k

k

(k − 1)ϵ2 2

P TV(P¯ll+t,Pˆll+t) ≥ ϵ ≤ 2M+2k · exp −

.

t=k−1

l=i+1

In other words, whenever

M + log(1/δ) ϵ2

k = Ω

,

we have TV(P¯ll+t,Pˆll+t) ≤ ϵ/3, for all l = i+1,...,i+k, and t = k −1,k, with 1−δ probability. Here Ω(·) means that we omit logarithmic order terms in M and ϵ.

| |
|---|

### F Detailed Setups for Offline and Online Experiments

Compute. All experiments run on a GPU cluster (Runpod) equipped with A100 (80GB) GPUs. GPU usage varies by method: Rebase and MCTS use two GPUs to separately serve their LLM and reward

Table 3: Hyperparameter configurations for certaindex. Rebase is evaluated on the MATH-OAI [39] subset of the MATH benchmark.

Algorithm Dataset Thres. (H˜τ) Thres. (Rτ) Detect @knob

SC MATH 0.7 / 5 SC GSM8K 0.7 / 5 SC LiveCodeBench 0.4 / 5

MCTS GSM8K 0.99 0.4 3 MCTS ASDiv / 0.4 3 Rebase GSM8K 0.85 0.99 16 Rebase MATH 0.75 / 16

models, while SC requires only one GPU for request processing. For final answer selection, SC uses simple majority voting, Rebase applies weighted majority voting, and MCTS selects the best path using a reward model via best-of-n.

General settings. Tables 1 and 2 summarize our offline and online evaluation workloads, respectively.

- Table 3 lists the certaindex hyperparameters for each workload, including the thresholds at the detection step (detect @knob). A program terminates at the detection step if its certaindex values meet all threshold conditions.

#### F.1 Batch Inference Setup: CoT

Metrics. For offline workloads, we measure tokens-to-accuracy, the total number of generated tokens needed to achieve specific accuracy levels across reasoning queries. This metric reflects the accuracy-cost trade-off, critical for end users as LLM platforms charge based on token usage. Table 1 details the offline experimental settings for SC/MCTS/REBASE. Each problem is allocated a maximum computation budget (parameterized by a Resource Cap) defining the sampling limit for SC and iteration limit for MCTS. Resource caps correspond to different points in Fig. 7. While scheduling methods may terminate programs early, they cannot exceed these caps. Query deadlines are calculated as the product of three factors: the SLO scale, the query’s difficulty factor, and a base deadline, reflecting the additional time required for more challenging problems.

We evaluate the Certaindex-based early termination method (§2.1) against baseline uniform token allocation across multiple scales of distilled DeepSeek models (7B, 14B, and 32B) [14] on mathematical reasoning benchmarks AIME24 and AMC23 [20], and MATH500 [39]. Unlike the baselines that uniformly increases token budgets, our appraoch early terminates by monitoring Certaindex at various intervals (T = 32, 64, 128. 256, and 320). All requests are given max token budget of 16K. For each interval, we vary the early termination parameter N (the required number of consecutive consistent answers), generating different points along each line. For fair comparison, appropriate accuracy thresholds were calibrated to model scale - with 32B models evaluated against stricter thresholds above QwQ [34] levels and reduced thresholds for smaller models - while setting higher targets for simpler tasks where greater accuracy is achievable.

#### F.2 Batch Inference Setup: SC, MCTS, Rebase

Metrics. For online workloads, we simulate query arrivals using dataset queries and assign deadlines to each query. System performance is evaluated by P90 deadline attainment, the percentage of queries completed within their deadlines. We vary arrival rates and SLO scales to assess system performance under different conditions. Tab. 2 outlines the online experiment settings. Request arrivals follow a Poisson process with varying rates. Deadlines are difficulty-aware, determined using a simple policy: oracle difficulty for each query is estimated through extensive trial runs (> 100) for each algorithm-dataset combination, classifying queries as always correct (difficulty factor = 1), always incorrect (= 3), or variable (= 2). Deadlines are calculated as the product of the SLO scale, the difficulty factor, and a base deadline, accounting for the additional time required for more challenging problems.

Baselines. For offline evaluation, we compare Dynasor against a modified SGLang intra-program scheduler using the following policies:

- • (1) baseline-even, allocates resources uniformly across all reasoning programs using the resource cap settings in Tab.1. Different cap values result in different accuracy levels, as larger caps allow more computation for all problems. Specifically, for each algorithm, all queries receive identical resources: SC uses the same number of branches, Rebase uses the same width, and MCTS uses the same number of search iterations, as specified in Tab.1. This baseline reflects the behavior of reasoning programs in current systems: user may allocate equal resources on all problems.
- • (2) baseline-length. This policy uses the cumulative token count generated at a specific step (specified as Detect@knob in Table 3) as the program’s progress signal. The detection step matches the one used in the certaindex-based approach. The scheduler either continues allocating resources or terminates the program based on a predefined token-length threshold.

Output token length is commonly used as a scheduling indicator in existing LLM serving systems [41; 46; 61]. It also correlates strongly with the compute requirements (knob size) as we have discussed in §B.3.

#### F.3 Online Inference Detailed Setup

Baselines. For online evaluation, we evaluate Dynasor against modified SGLang with different inter-program schedulers on P90 deadline attainment.

- • SGLang [35]. SGLang represents a strong baseline as it incorporates key optimizations in LLM serving: its longest-prefix matching (LPM) algorithm efficiently batches requests with common prefixes, while prefix caching reduces redundant computation. We enabled these optimizations and tuned for stable performance with 70% memory utilization.
- • Parrot [40]. Parrot uses gang scheduling (App-FIFO) to prioritize requests in the same program. We implemented its scheduler on top of SGLang for fair comaprison. By grouping requests from the same program together, it minimizes context switches and maximizes KV cache reuse across batches.

For experiments (a) and (b), we use fixed resource cap settings (SC: 20, MCTS: 15, Rebase: 128) to compare Dynasor with baselines, which lack adaptive compute scheduling for LLM reasoning programs. We measure SLO attainment while ensuring accuracy is maintained. Our method employs the same resource allocation policy validated in §4.1 and §H, which preserves accuracy. In experiment (c), we explore the relationship between achievable accuracy and SLO attainment by varying resource cap settings.

### G Other Ablation Studies

#### G.1 Effectiveness of CoT in Deepseek-R1

To validate scalability, we extended our experiments to the larger DeepSeek-R1 model on AIME and AMC datasets (Figure 17). The results align with our findings from smaller distill models, demonstrating consistent efficiency gains: DeepSeek-R1 achieves 12% token savings on AIME problems and 24% on AMC problems while maintaining baseline accuracy levels.

1.00

| | | | |
|---|---|---|---|
| | | | |
| |-1|2%| |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | |-|24%| | |
| | | | | | |
| | | | | | |

0.76

0.98

0.74

Accuracy

Accuracy

0.96

AIME

AMC

0.72

0.94

0.70

0.92

0.68

0.90

6000 7000 8000 9000 Tokens

3000 3500 4000 4500 5000 Tokens

Baseline

Ours (64)

Ours (256) Ours (320)

Ours (32)

Ours (128)

Figure 17: Apply Dynasor on DeepSeek-R1

#### G.2 Ablation: Scheduling Component Contributions

Figures 18 and 19 show the contribution of different components using SC on GSM8K and MATH datasets, measuring mean latency and maximum sustainable rate under the same P90 attainment. Using LPM (SGLang) as the baseline, we analyze three factors: gang scheduling, SJF, and certaindexbased resource management. Gang scheduling improves latency and throughput in both uniform and certaindex-aware resource allocation by ensuring the requests in the same program are scheduled together, reducing resource fragmentation and context switching overhead. For GSM8K workload, certaindex-based resource management dominates the improvements, reducing mean latency by up to

Uniform Allocation

Ours

| |41|0s| |
|---|---|---|---|
| | |30|6s|
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| |16|5s| | |
| | |15|9s 13|4s|
| | | | | |

Uniform Allocation

Ours

400

400

MeanLatency

- 0
- 1
- 2

- 0
- 1
- 2

Rate:Program/s

| | | | |
|---|---|---|---|
| |1.0|x 1.1|x|
| | | | |
| | | | |

| | | |1.9|x|
|---|---|---|---|---|
| |1.2|x 1.3|x| |
| | | | | |
| | | | | |

200

200

0

0

LPM Gang

Ours (LPM) Ours (Gang) Ours (SJF)

Figure 18: Performance improvement breakdown in online SC (GSM8K): Impact of gang scheduling, certaindex-based allocation, and SJF on mean latency with fixed request rate (rps = 16).

LPM Gang

Ours (LPM) Ours (Gang) Ours (SJF)

Figure 19: Performance improvement breakdown of online self-consistency (MATH) under fixed P90 SLO constraints.

60% through token savings of up to 50%, while Gang scheduling and SJF contributions are more modest. For MATH workload, certaindex-aware allocation achieves a 1.2x peak rate through early termination of low-value computations (as only 10% tokens are saved on MATH), while SJF provides the most significant gain with a 1.9x peak rate by prioritizing shorter jobs and reducing head-of-line blocking. These results validate our two-level scheduling approach: intra-program optimization through gang scheduling and certaindex-aware allocation, combined with inter-program optimization through SJF.

#### G.3 Fairness Analysis

We apply finish-time fairness (§D.3.1) as the fairness metric and use latency / number of tokens as a proxy. Figure 20 shows the comparison on certaindex-based resource scheduling, gang scheduling, and SJF on finish-time fairness on MATH using SC with a rate of 8 pps using Llama3.1 8B. Gang scheduling shows consistent fairness improvement compared to non-gang scheduling. This shows that prioritizing existing programs can improve finish-time fairness. Certaindex-based resource allocation also consistently improve finish-time fairness due to resource cutting by the intra-program scheduler. Adding SJF shows fairness improvement at the later 50% fraction of the job compared to without SJF, and at the later 35% fraction of job compared to LPM. In all case, SJF shows the fairness metric no worse than even resource allocation.

CDF of Finish Time Fairness (Latency / #Tokens)

1.0

FractionofJobs

Ours (SJF)

Ours (Gang)

0.5

Ours (LPM)

SGLang

Parrot

0.0

0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 Finish Time Fairness (Latency / #Tokens)

Figure 20: Finish-Time Fairness.

#### G.4 Fine-grained Resource Allocation

We compare simple thresholding against fine-grained resource allocation in this section. §4.1 and §4.2 adopt a simple static threshold mechanism: extract certaindex at a fixed step and decide if to terminate program based of the value of certaindex. While effective, there is room for optimization through more sophisticated scheduling strategies. To explore this, we conducted experiments in offline batch processing using the setting (SC, MATH) and (MCTS, ASDiv), both with a maximum step limit of 20 as per-query resource cap, to evaluate more complex allocation strategies that enable early termination while maintaining accuracy. The results, measured in token savings, are presented in Tab. 4. For the static threshold, we collected certaindex and apply the threshold in Tab. 3.

- Table 4: Token consumption comparison of different scheduling strategies while maintaining accuracy Allocation Method SC/MATH MCTS/ASDiv

|Baseline|1.180M<br><br>|354K|
|---|---|---|
|Static Thres. (Ours)<br><br>|1.05M (-11.0%)|308K (-13.0%)|
|+ Initial Step Curve Fit.(Ours)|1.04M (-11.9%)<br><br>|306K (-13.6%)|
|+ 5-Step Thres. (Ours)<br><br>+ Single-Step Thres. (Ours)<br><br>+ Dynamic Curve Fitting (Ours)|1.03M (-12.7%) 1.03M (-12.7%) 1.01M (-14.4%)<br><br>|307K (-13.3%) 306K (-13.6%) 298K (-15.8%)<br><br>|

We first examine an Initial Step Curve Fitting approach, which fits a skyline curve (illustrated by the green lines in Fig. 12) using certaindex values collected at the same steps as the static threshold. Although this method enables finer-grained resource allocation for varying certaindex values, it relied on early certaindex values, which maybe inaccruate as reasoning progresses, resulting in marginal improvements (less than 1% compute savings) compared to simple static thresholding.

To enhance prediction accuracy, we test more frequent certaindex collection: every 5 steps (5-Step Thres.) and every step (Single-Step Thres.), based on which we similarly apply threshold filtering or skyline curve fitting at every step (Dynamic Curve Fitting). These approaches achieve higher token savings (up to 3.4% over static threshold). However, implementing them introduces scheduling trade-offs in real-world deployment due to their impact on scheduling and parallelism. Specifically, frequent certaindex collection may disrupt the concurrent execution of reasoning programs. For instance, in SC, instead of running 20 samples concurrently, we must process them in sequential batches (e.g., 4 batches of 5 samples) to track the certaindex. This shift from parallel to sequential execution substantially increases latency. Our benchmark shows an increase from 289s to 366s in mean latency when serving 500 programs. Given these practical constraints, we opt to implement the simple static threshold in our end-to-end experiments, prioritizing system performance over marginal token savings; but note in applications which prioritize cost over latency, such allocation strategies remain effective and are implemented in Dynasor.

### H Token-to-accuracy Performance of Dynasor on MATH using SC

Figure 21 presents the token-to-accuracy results using a Llama3.1 8B Instruct model, where resources are allocated by Dynasor using the threshold and detect @knob configurations described in Tab. 3. Our proposed method achieves the same accuracy while reducing computational costs by 11%.

Ours Baseline-even

Self-Consistency MATH

0.56

Accuracy(%)

0.54

0.5 0.8 1.0 Number of Tokens (1e7)

Figure 21: Token-to-accuracy Performance Using SC on MATH

