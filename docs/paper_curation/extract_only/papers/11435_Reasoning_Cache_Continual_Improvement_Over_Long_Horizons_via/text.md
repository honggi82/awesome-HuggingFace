## Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL

###### Ian Wu1, Yuxiao Qu1, Amrith Setlur1 and Aviral Kumar1 1Carnegie Mellon University

70

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | |+5.0| | |
|S|OTA 4B R|easoning|Model (256|k)| |
| | | | | | |
| | | | |+30.2| |
| | | |Base|+ RC| |
|B|ase Mode|l (16k)<br><br>|RCT-4|B + RC| |
| | | | | | |
| | | | | | |

sample summary

Acc.onHMMT'25

65

Training Rollouts

reasoning summ

60

#### arXiv:2602.03773v2[cs.LG]22Mar2026

###### LLM LLM

###### LLM

###### LLM

…

55

50

LLM

RL

input

45

generate rollouts

40

generate

summarize

generate

16 144 272 400 512

Reasoning token budget (k)

RC Decoding

RC Training

Figure 1: Left: Illustration of the RC algorithm. RC decoding replaces standard autoregressive decoding at both train and test time. During RC decoding, the LLM generates a reasoning trace, summarizes it, discards the original trace, and conditions subsequent reasoning on this summary. This design decouples the effective reasoning horizon from the length of any single reasoning trace, thus maintaining tractable rollout lengths for outcome-reward RL while also enabling extrapolation at test time. Right: Performance on HMMT 2025 (November) vs. reasoning token budget. Our RC-trained model RCT-4B (blue, trained from Qwen3-4B-Instruct-2507 at 16k train budget) extrapolates to outperform both the base model with RC decoding (green) and the specialized Qwen3-4B-Thinking-2507 reasoning model (evaluated at 256k test tokens).

Abstract: Large Language Models (LLMs) that can continually improve beyond their training budgets are able to solve increasingly difficult problems by adapting at test time, a property we refer to as extrapolation. However, standard reinforcement learning (RL) operates over fixed problem distributions and training budgets, which limits extrapolation amidst distribution shift at test time. To address this, we introduce RC, an iterative decoding algorithm that replaces autoregressive decoding during both training and inference. RC exploits an asymmetry between the response generation and summarization capabilities of LLMs to construct reasoning chains that consistently improve across iterations. Models trained to use RC can extrapolate and improve over reasoning horizons more than an order of magnitude longer than those seen during training. Empirically, training a 4B model with RC using a 16k-token training budget improves performance on HMMT 2025 from 40% to nearly 70% when evaluated with 0.5m test-time tokens, outperforming both comparably sized models and many larger reasoning LLMs. We also show that models trained with RC can more effectively leverage existing scaffolds to further scale test-time performance, due to the improved summary-conditioned generation abilities learned through training.

##### 1. Introduction

Large language models (LLMs) exhibit the ability to solve complex problems by generating long reasoning traces at test time. As LLMs become more capable, we naturally expect them to be able to solve harder tasks by reasoning for longer, even without external supervision. This expectation mirrors human cognition: humans continually improve their reasoning by revisiting earlier conclusions and reallocating effort to discover new information in new ways over the course of problem solving. We would like our models to behave similarly, such that additional reasoning at test time translates to improved performance over horizons much longer than any single reasoning step. This underlies the notion of in-context exploration [33], which learns an implicit algorithm or “procedure” for allocating test-time compute such that spending more computation systematically improves outcomes. If learned robustly, in-context

Corresponding author(s): ianwu@andrew.cmu.edu Code repository: github.com/IanYHWu/rc

exploration should enable models to continually improve over long horizons at test time, perhaps over hours or even days, and across millions of tokens, all without modifying the model weights.

However, current LLM training paradigms limit the forms of in-context exploration that can be learned. Supervised fine-tuning (SFT) teaches models to imitate the content of reasoning traces, rather than learning the algorithm that generates them [4, 32]. This inhibits the learning of systematic reasoning procedures and thus the ability to perform effective in-context exploration. While reinforcement learning (RL) incentivizes models to learn reasoning procedures rather than mere imitation [38, 47], RL is typically applied over a fixed prompt distribution and a bounded training rollout length. As a result, models are optimized to exploit this finite budget, rather than to extrapolate beyond it. When such models encounter complex problems that require more reasoning to solve, two failure modes emerge. First, they may prematurely terminate within their training budgets and fail to make progress by reasoning for longer. Second, when models continue beyond this budget, distribution shift may occur as generation proceeds from conditional distributions that differ substantially from those seen during training; reasoning traces in this regime are often repetitive and verbose [19, 33]. This raises a natural question: how can we train models to extrapolate their reasoning far beyond their training configurations?

To answer this question, we introduce Reasoning Cache (RC), an iterative decoding algorithm that replaces standard autoregressive decoding during both training and inference. In RC, the model generates a reasoning trace, summarizes it (the “cache”) and discards the original trace over multiple turns, with subsequent generation conditioning only on the previous summary rather than the full reasoning trace. For training, we introduce an RL approach that teaches the model to exploit RC by improving its summary-conditioned reasoning capabilities. Our approach combines typical on-policy RL with off-policy learning via a replay buffer that enables reuse of cached summaries, allowing the model to train over long effective horizons without generating prohibitively long training rollouts. The design of the decoding process used by RC is motivated by two key observations. First, iterative decoding provides explicit control over test-time compute: we can scale the number of reasoning iterations while keeping each step within the training distribution, mitigating distribution shift as effective reasoning length grows. Second, making such iteration effective requires consistent progress across steps. RC achieves this by exploiting a summarization–generation asymmetry: models are often better at summarizing prior reasoning and reasoning from summaries than at producing correct solutions from scratch. Our RL objective explicitly amplifies this asymmetry, enabling larger improvements across successive iterations.

Empirically, models trained with RC exhibit strong and consistent extrapolation ability. While RC decoding alone is already effective when the base model can reliably follow instructions and reason from summaries, RC-trained models extrapolate substantially further. On the mathematical reasoning benchmarks HMMT 2025 (November) and IMO-AnswerBench, our RC-trained RCT-4B model substantially outperforms both Qwen3-4B-Instruct-2507 (the base model) and Qwen3-4B-Thinking-2507 (a much stronger reasoning model) through extrapolation. For example, accuracy on HMMT 2025 improves from 40% to 70% as the test budget is scaled from 16k to 512k tokens (Figure 1, right), while on IMO-AnswerBench [20], performance rises from 41% at 16k tokens to 58% at 256k tokens, surpassing larger models such as Qwen3-235B, Qwen3-30B-A3B-Instruct-2507 and Nemotron-3-Nano-30B-A3B using standard autoregressive decoding, despite training only at a 16k budget. Notably, despite having been trained solely on mathematical reasoning data, RCT-4B also achieves substantially higher performance than the base model on the FrontierScience [24] scientific reasoning benchmark when extrapolated to 256k tokens, suggesting that RC induces transferable algorithmic behavior rather than domain-specific knowledge. Finally, we show that RCT-4B is consistently better at exploiting a variety of test-time scaffolds than both

the base model and a base model post-trained with standard RL, with and without RC decoding. This indicates that optimizing for summary-conditioned generation yields models that are not only better at leveraging RC decoding, but are also generally more effective at using external context to guide reasoning.

##### 2. Related Work

Long-horizon reasoning by scaling test-time compute. Increasing the length of reasoning traces is a fundamental approach to scaling test-time compute [9, 42], as longer traces allow for more extensive in-context exploration that can substantially improve performance [7, 33, 47]. Although prior work finds that on-policy RL is effective at teaching LLMs to generate longer autoregressive responses [9], scaling in this way suffers from a fundamental limitation: models cannot reliably extrapolate much beyond the reasoning lengths seen during RL training [19, 33]. Our work therefore aims to develop methods that enable extrapolation, rather than merely scaling reasoning lengths by increasing training budgets.

Test-time extrapolation of reasoning. Prior work attempts to enable extrapolation mainly through one of two approaches. The first modifies training via carefully designed datasets and curricula to encourage in-context exploration [2, 19, 33]. Although this has been shown to enable extrapolation to around 3–4× the training budget, performance typically saturates beyond this. The second approach modifies the RL reward structure to implicitly encourage extrapolation (e.g., via dense rewards that credit intermediate reasoning segments [29]) but retains pure autoregressive decoding at inference time. In both cases, in-context exploration behaviors are implicitly learned through free-form autoregressive generation and thus remain coupled to length budgets in the training setup. When test-time conditional distributions fall outside the training support, as is seen when test lengths greatly exceed train lengths, these approaches do not improve further and instead result in verbose and repetitive behavior [19, 33].

Iterative decoding for scaling test-time compute. Prior work explores prompting LLMs to iteratively transform their own outputs to scale test-time compute, ranging from simple self-correction [12, 13] and self-refinement [21, 37] to more complex scaffolds that combine iterative and parallel compute [35, 40]. Others consider training models to apply transformations more effectively rather than relying on prompting alone [14, 26]. Such transformations may occur over multiple time steps: Venkatraman et al. [40], for example, train models to iteratively aggregate parallel reasoning traces, while Madaan et al. [22] consider distilling parallel reasoning traces into summaries that are conditioned on for final answer generation. In contrast to these works, RC scales test-time compute sequentially through extrapolation rather than in parallel. This is reflected in our training procedure, which directly optimizes for multi-step reasoning that can extend to very long horizons through the use of a summary replay buffer. Furthermore, we find that RC training enables models to better exploit other scaffolds by teaching them to more effectively leverage external guidance, thus making our approach complementary to many existing works.

Memory in multi-turn interaction. RC summaries can be viewed as compressed memory states that are updated as the policy acts over iterations. Prior work primarily consider using similar memory states to store external context (e.g. retrieved web pages, user responses etc.) that is dynamically recalled in later steps, often as part of multi-turn question-answering or conversation systems [16, 49]. Our method instead uses memory states to store abstractions of self-generated reasoning traces for solving reasoning problems, an approach that has been explored using prompting-based approaches [10, 39, 43]. Unlike these works, we focus on training the model to better utilize these memory states for downstream reasoning, which we show yields significant improvements over prompting-only methods.

##### 3. Preliminaries and Notation

Consider a policy 𝜋𝜃(·|𝑥) that generates tokens autoregressively conditioned on 𝑥. At test time, the policy is given a token budget and allocates this to reason. Our interest is test-time performance as a function of the test token budget, particularly with large test-time budgets.

Standard RL training for LLM reasoning. Let 𝒟train denote a training distribution of prompt-answer pairs (𝑥,𝑦). On-policy reinforcement learning (RL) optimizes the expected reward of the policy:

E𝑥,𝑦∼𝒟train [︁E𝑧∼𝜋𝜃(·|𝑥)[𝑟(𝑦,𝑧)]]︁, s.t. |𝑧| ≤ 𝐻train. (Training objective) (3.1)

max𝜋

𝜃

Here, 𝑧 denotes an on-policy rollout autoregressively sampled from 𝜋𝜃. The rollout encodes a reasoning trace and is generated within a fixed training budget 𝐻train. The reward function 𝑟(𝑦,𝑧) evaluates the correctness of the rollout, typically by extracting the final answer from 𝑧 and comparing it against the ground-truth label 𝑦. To solve this optimization problem, we can use outcome-reward policy-gradient methods: one common choice is GRPO [34] (see Appendix H), which we use throughout this work.

Test-time extrapolation of LLM reasoning. Equation 3.1 optimizes performance only over the empirical distribution of training prompts 𝒟train, and only under a fixed 𝐻train. At test time, we may want to ensure that our trained model attains high accuracy on a different prompt distribution and under a larger budget:

TestPerf(𝜋𝜃) def:= E𝑥,𝑦∼𝒟test [︁E𝑧∼𝜋𝜃(·|𝑥)[𝑟(𝑦,𝑧)]]︁, s.t. |𝑧| ≤ 𝐻test (Test-time objective) (3.2)

where 𝐻test is the test budget; in general, the training and test distributions differ (i.e. 𝑝train(𝑥) ̸= 𝑝test(𝑥) and 𝐻test ≫ 𝐻train). When a model trained to optimize Equation 3.1 can leverage a larger test budget to achieve TestPerf(𝜋𝜃)|𝐻test > TestPerf(𝜋𝜃)|𝐻train, we say that it can extrapolate.

##### 4. Problem Statement

Does optimizing performance at 𝐻train (Equation 3.1) also optimize extrapolation at 𝐻test (Equation 3.2)? Unfortunately, the answer is no. During training, the model receives positive reward only

for rollouts that terminate within 𝐻train tokens. This implicitly penalizes longer rollouts and encourages “premature” termination near 𝐻train at test time. Moreover, when the model does continue beyond 𝐻train at test time, it must effectively operate on the sorts of conditional distributions it was never trained on. While this form of distribution shift is not problematic if the model has learned a true “operator” [29] that enables the chaining of behaviors to solve problems, it is unclear whether RL can learn such operators from a finite, fixed prompt set. We instead circumvent this challenge by altering the decoding algorithm such that the model never encounters significant distribution shift even when reasoning at long horizons.

Why do we need extrapolation? Can we simply increase 𝐻train to match 𝐻test during RL training? Doing so would obviate extrapolation altogether. However, there are two main problems with this approach.

First, any new test distribution we encounter may contain harder problems requiring 𝐻test ≫ 𝐻train to solve, and so it would be better to train models that can adapt to larger test budgets on-the-fly (i.e., training models to “continually adapt”). Second, memory and compute costs, as well as the effectiveness of RL training, scale aggressively with response length, making long-horizon on-policy RL prohibitively expensive and challenging. These challenges indicate that we cannot just scale 𝐻train, and should instead train models to extrapolate as more compute is provided.

Problem Mark writes the squares of several distinct positive integers (in base $10$) on a blackboard. Given that each nonzero digit appears exactly once on the blackboard, compute the smallest possible sum of the numbers on the blackboard.

Reasoning We are given a problem: Mark writes the squares of several distinct positive integers (in base 10) on a blackboard. Given…

Summary In this solution, the goal is to find the smallest possible sum of distinct perfect squares such that the digits in all the squares together contain each nonzero digit (1 through 9) exactly once, with no digit repeated and no zeros counting toward the nonzero digit constraint. After extensive trials of various combinations of small squares, the only valid configuration found that satisfies all conditions is: the squares of 1, 3, 5, 6, and 28 (i.e., 1² = 1, 3² = 9, 5² = 25, 6² = 36, and 28² = 784)...

Summary-Conditioned Reasoning We are to **find the smallest possible sum** of **distinct perfect squares** such that when we write down **all digits** of those squares, we get **each nonzero digit from 1 through 9 exactly once**, and zero digits are ignored (do not count). Our task is to **minimize the sum** of the squares such that their digits (ignoring 0s) together use each digit 1–9 exactly once. Let's first

**analyze and verify** the previous attempt. They proposed: $1^2 = 1$ → digit: 1…

- Figure 2: Illustrative example of RC’s output. The model generates reasoning given an input problem, before conditioning on the reasoning to generate a summary between 1-2 paragraphs long. The model then conditions on the summary to generate new reasoning. As we show later, RC training improves the ability of the model to perform summary-conditioned reasoning, which enables the model to continually improve over long horizons. See Appendix M for a full example of an RC output.

##### 5. Enabling Extrapolation with Reasoning Cache

Our goal is to develop a method that trains models under a fixed token budget and a finite prompt set, while still allowing them to extrapolate beyond the training horizon. To achieve this, we replace autoregressive decoding with an iterative decoding algorithm Alg(𝜋𝜃;𝑥) during training and inference. This algorithm, Alg, leverages the structure of reasoning along with asymmetries present in LLMs to support long-horizon reasoning at test time while remaining amenable to training under a much smaller 𝐻train. We begin by concretizing the key desiderata that Alg should satisfy.

Choosing an effective decoding algorithm. An effective choice of Alg must satisfy two desiderata. First, it should define an iterative procedure in which the number of iterations monotonically controls test-time compute, while each iteration operates on conditional distributions that remain close to those encountered during training. An Alg satisfying this desideratum avoids the main limitations of autoregressive decoding in standard RL: (1) by enabling longer responses through increasing the iteration limit, it mitigates premature termination biases induced by fixed-length RL training; and (2) by restricting autoregressive generation within each iteration to at most 𝐻train tokens, it reduces train–test distribution shift even when the effective reasoning horizon is much larger. Second, the algorithm should retain expressivity comparable to autoregressive decoding, allowing each iteration to refine or extend the current rollout and explore new directions. An Alg that satisfies this property can make consistent progress across many iterations, enabling continual extrapolation over long horizons.

###### 5.1. RC: A Multi-Turn Decoding Algorithm

We now introduce a decoding algorithm that satisfies these desiderata. Our algorithm, which we call Reasoning Cache (RC), is an iterative decoding approach that alternates between response generation and summarization. Being an iterative decoding algorithm, RC naturally fulfills our first desideratum:

| | |4B| | | | |
|---|---|---|---|---|---|---|
| | |30|B| | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Acc.onHMMT'25

Acc.onHMMT'25

55

4B + RC

65

Tokensused(k)

100

60

4B + Self-Refine

80

50

55

4B + Self-Verify

60

50

45

4B

40

45

30B

20

40

40

0

16 64 112 160 192

16 64 112 160 192 Reasoning token budget (k)

16 64 112 160 192

- Figure 3: Left: Accuracy vs. test-time token budget. RC decoding improves performance as token budget 𝐻test is increased far beyond 𝐻train = 16k. Middle: Total tokens used vs. test-time token budget. Total reasoning tokens used by RC increases linearly as we increase the reasoning token budget. Shaded regions indicate the 5th-95th percentile; lines indicate the mean. Right: Accuracy vs. token budget for iterative decoding methods. RC is a more effective method for enabling extrapolation than self-verification and self-refinement, highlighting the benefits of exploiting the summarization-generation gap.

we can increase test-time compute by increasing the number of summarization-generation turns, while also avoiding significant shifts in the conditional distributions encountered at each turn by only ever autoregressively generating at bounded lengths 𝐻train ≪ 𝐻test. To satisfy our second desideratum, RC relies on two properties of LLMs. First, reasoning traces contain redundant tokens: many tokens encode steps that are useful for local progress but need not be retained verbatim to guide future actions. This allows us to discard a significant portion of tokens (e.g. via summarization) so long as key information is retained. Second, as we consistently find in our experiments, base LLMs often exhibit summarizationgeneration asymmetry, in that producing a correct response conditioned on a summary of a previous attempt is easier than generating a correct response from scratch; this asymmetry arises from the instruction-following abilities of LLMs, which allows them to use summaries of prior generations to guide subsequent reasoning. RC exploits this by periodically compressing responses into a cache and conditioning subsequent generation on it, allowing the model to refine, extend, or restart reasoning across iterations as needed. See Figure 2 for an example of RC’s outputs.

Let 𝑥 denote the prompt and let 𝑡 ∈ N index the decoding turn. RC maintains: (1) a reasoning trace 𝑧𝑅(t) and (2) a summary 𝑧𝑆(t), with 𝑧𝑆(0) initialized to the empty string. At each turn, 𝑧𝑅(t) is generated under a fixed token budget 𝐻𝑅, while 𝑧𝑆(t) is generated under 𝐻𝑆 ≪ 𝐻𝑅. Decoding proceeds by alternately prompting the base model with two distinct system instructions ℐ𝑅 and ℐ𝑆 (see Appendix N):

𝑧𝑅(t) ∼ 𝜋𝜃(︂· | ℐ𝑅,𝑥, 𝑧𝑆(t−1) )︂, (5.1)

𝑧𝑆(t) ∼ 𝜋𝜃(︂· | ℐ𝑆,𝑥, 𝑧𝑅(t) , 𝑧𝑆(t−1) )︂. (5.2)

ℐ𝑅 instructs the model to generate reasoning conditioned on the current cache, while ℐ𝑆 instructs the model to compress the current reasoning trace and previous summary into an updated summary that encodes high-level information about the strategies employed and conclusions reached in previous turns.

After T turns, the final output is given by 𝑧 := 𝑧𝑅(T). We denote this iterative process (see Figure 1) as:

( 𝑧𝑅(1) , 𝑧𝑆(1) ,..., 𝑧𝑆(T−1) , 𝑧𝑅(T) ) ∼ Alg(𝜋𝜃;𝑥). (5.3)

Extrapolation with RC. Because each step is allocated a fixed budget 𝐻𝑅 + 𝐻𝑆, the total effective budget under RC is T×(𝐻𝑅×𝐻𝑆). Since 𝐻𝑅 ≫ 𝐻𝑆, we drop 𝐻𝑆 and approximate the budget as T×𝐻𝑅 def:= 𝐻test. If performance improves in the regime T′ × 𝐻𝑅 ≫ 𝐻train, we say that RC enables extrapolation.

16

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Acc.onHMMT'25

Acc.onHMMT'25

56

50

12

Acc.gain

52

40

HR = 4k HR = 8k HR = 16k

Inst, Inst

8

48

Think, Inst

30

4

44

Think, Think

20

0

40

0 64 128 192 256 320 384 Additional reasoning budget (k)

4 52 100 148 192 Reasoning token budget (k)

Ans.1-2sen.1par.2par.Mul.par.Full

- Figure 4: Left: Accuracy at different levels of summary detail. All accuracies measured at 𝐻test = 192𝑘; red dotted line indicates base model performance without RC. Performance degrades when summary detail is either too high or too low. Middle: Relative accuracy improvement (over standard autoregressive decoding) on HMMT 2025. Replacing Qwen3-4B-Instruct2507 with Qwen3-4B-Thinking-2507 for summarization (Think, Inst) reduces gains. Using Qwen3-4B-Thinking-2507 for both

tasks (Think, Think) further reduces gains. Right: Accuracy vs. test-time token budget with various 𝐻𝑅. Reducing 𝐻𝑅 from 16k to 8k leaves RC performance unchanged, whereas decreasing it further to 4k negatively impacts performance.

###### 5.2. Experimental Evaluation

Experimental setup. We now validate whether LLMs possess the ability to to utilize RC without additional training. We evaluate RC decoding with Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct2507 [30], two hybrid LLMs capable of both complex reasoning and instruction-following (see Appendix D for similar results from another model family). Using these LLMs as our base models, we run RC decoding for T = 12 turns with 𝐻𝑆 = 2048 and 𝐻𝑅 = 16k, giving us a total budget of 𝐻test = 192k. This is far larger than both models’ 𝐻train, which we estimate to be about 16k (see Appendix C for evidence justifying this). We use the November version of HMMT 2025 as our evaluation dataset (this competition was conducted after the base models were released), and generate 16 RC outputs per problem.

- Finding 1: RC enables extrapolation. We plot how accuracy evolves with the token budget 𝐻test in

- Figure 3 (left). We find that RC extrapolates reasoning far beyond 𝐻train = 16k: 4B model accuracy increases by 17% as the test token budget is scaled from 16k to 192k, while 30B model accuracy increases by 12%. We also plot how actual token usage varies with reasoning token budget in Figure 3 (middle). We find that the cumulative number of tokens used scales linearly with the provided budget, which indicates that the model utilizes additional test time compute as provided and does not substantially shorten its responses at later iterations. Overall, our findings demonstrate that RC satisfies both desiderata outlined in Section 5 and thus enables effective extrapolation.

- Finding 2: Summary-based abstractions are key to effective extrapolation. We examine the role of summarization–generation asymmetry by experimenting with iterative decoding methods that do not utilize summarization. Concretely, we remove the summary step and instead condition each iteration

directly on the full response from the previous iteration 𝑧𝑅(t), and prompt Qwen3-4B-Instruct-2507 (𝐻𝑅 = 16k) to either verify-then-correct (self-verify) or self-refine its solution (details in Appendix J). Figure 3

(right) shows that RC consistently outperforms these baselines across all 𝐻test values, demonstrating that summary-conditioned generation provides benefits over other iterative methods. We attribute this to two factors: first, conditioning on summaries keeps context lengths bounded and in-distribution, whereas iterating on raw responses exceeds 𝐻train and induces distribution shift. Second, summarization serves to remove redundant “distractor” tokens [11, 17] that obfuscate key findings and other important information, thereby yielding clearer guidance for subsequent reasoning.

- Finding 3: Summary detail level matters. Next, we study how much information summaries should retain. We vary the prompt ℐ𝑆 to produce summaries of differing detail, ranging from answer-only to

multiple paragraphs (Figure 21), and compare these to our default approach and to full-trace conditioning (self-refinement). Figure 4 (left) reports the accuracy of Qwen3-4B-Instruct-2507 at 𝐻test = 192k across various choices of ℐ𝑆. Performance degrades with very short summaries, improves as more detail is added, and peaks with ≥2-paragraph summaries; omitting summarization entirely degrades performance.

- Finding 4: Base models must be good instruction-followers for RC to be effective. We replace Qwen3-4B-Instruct-2507 with the specialist reasoning model Qwen3-4B-Thinking-2507, which excels at reasoning but possesses weaker instruction-following abilities and thus a less clear summarization-generation asymmetry. We evaluate using the reasoning model only for summaryconditioned generation (“Think, Inst”) and for both summary and summary-conditioned generation (“Think, Think”) (𝐻𝑅 = 64k).

- Figure 4 (middle) shows that “Think, Inst” and “Think, Think” both achieve smaller relative gains; qualitative inspection reveals that the reasoning model sometimes ignores summaries during generation and omits key details during summarization. However, note that RC still provides positive performance gains in all configurations, suggesting that sufficient asymmetry is present for training to potentially be effective. Indeed, LM-Provers et al. [18] demonstrate that RC training substantially improves the proofwriting capabilities of Qwen3-4B-Thinking-2507, confirming that sufficient summarization-generation asymmetry exists even in specialist reasoning models for the method to be effective.

- Finding 5: Reducing 𝐻𝑅 by too much degrades performance. By default, we set 𝐻𝑅 = 𝐻train = 16k, and since LLMs rarely generate traces longer than 𝐻train, we only consider decreasing it. Figure 4 (right) shows that reducing 𝐻𝑅 from 16k to 8k has minimal impact despite the fraction of incomplete traces increasing from 0% to 20% (see Figure 5). Reducing 𝐻𝑅 further to 4k causes nearly 50% of traces to terminate early, this time substantially degrading performance. This implies that while 𝐻𝑅 can be decreased, it must still be large enough for redundancy to emerge. When 𝐻𝑅 is set too small, the resulting summaries capture only shallow progress that provides insufficient signal for continuation, which encourages the model to restart reasoning from scratch instead.

100

Term.Rate(%)

80

60

40

20

0

0 2 4 6 8 10 12 14 16 Length in tokens (k)

Figure 5: Termination rates of Qwen34B-Instruct-2507 with RC as a function of length. Measured on HMMT 2025 and across all T = 12 turns. Virtually all reasoning traces terminate within 16k tokens, suggesting that the model is not trained to reason beyond this. We consider a trace to have terminated after it generates boxed{}.

###### 5.3. Analysis of Summary-Conditioned Generations

80

4B

Perc.ofSamples

We analyze the content of summary-conditioned generations produced by RC and find that they commonly exhibit three high-level strategies: (1) verification, where the model generates reasoning to explicitly verify intermediate or final results stated in the summary; (2) exploration, where the model deliberately pursues a different strategy from that used in the summary; and (3) refinement, where the model acknowledges the summary and repeats the same strategy without attempting verification or exploring alternatives. To quantify the prevalence of these behaviors, we extract summaries and their subsequent reasoning traces and pass them to an LLM-based annotator (Figure 22), which assigns each sample to one of the three categories above, as well as a none category when no references to the

30B

60

40

20

0

Verify Explore Refine None

Figure 6: Percentage of RC reasoning traces that employ various reasoning strategies. Verification of previous reasoning is the most common strategy, followed by exploration of new strategies.

store summary in buffer

sample summary

reasoning summ

Training Rollouts

###### LLM LLM

###### LLM

###### LLM

# …

B

LLM

RL

sample summary

generate rollouts

generate

summarize

generate

- Figure 7: Illustration of RC rollout generation for training with a replay buffer ℬ. We sample an input and a summary from ℬ, and use this as the starting point to run Ttrain steps of RC decoding. New summaries generated through this process are stored in ℬ, replacing older summaries corresponding to the same input problem. We then sample from the Ttrain newly generated summaries, and condition on these to generate training rollouts that are optimized via Equation 6.1.

summary are made and reasoning restarts from scratch. Figure 6 shows that the model relies heavily on summaries to guide subsequent generations, with very few samples classified as none. The most common strategy is verification, although a substantial fraction of samples also exhibit exploration and refinement.

Key Takeaways: Inference-only experiments with RC

• RC enables extrapolation even without RC-specific training. It exploits the summarizationgeneration asymmetry and is thus most effective when paired with instruction-following models.

##### 6. Training Models to Extrapolate with RC Decoding

Having established the design of Alg, we now describe our method for training models to use it. Our analysis in Section 5.2 shows that RC extrapolation depends on the model’s ability to iteratively reason from and improve upon summaries of past iterations. Accordingly, our training objective is to strengthen summary-conditioned generation: given a problem and a summary, we should train our model to generate improved reasoning that is more likely to yield a correct answer. The iterative structure of RC makes this objective amenable to standard outcome-reward RL as we can run RC for multiple iterations and apply gradient updates independently at each turn. By setting 𝐻𝑅 to a “typical” value for the base model, such that most responses end with a final answer, we can assign outcome-based rewards at each step and avoid credit assignment across iterations. While this approach is indeed myopic, we note that the iterative nature of RC decoding ensures that this does not reinforce the resulting bias of premature termination.

Formally, we run RC for Ttrain turns for each problem 𝑥 in a training batch. We collect the summaries generated from this 𝑧𝑆 := (𝑧𝑆(1),...,𝑧𝑆(Ttrain)) and uniformly sample 𝑁summ ≤ Ttrain unique summaries per problem. We then generate 𝐾 reasoning traces conditioned on each sampled summary, assign outcome rewards and compute advantages over these samples:

E𝑥,𝑦∼𝒟train, 𝑡∼𝑈[1,Ttrain] [︂E𝑧′∼𝜋

]︂,

𝜃(·|𝑥,𝑧𝑆(𝑡)) [︀𝑟(𝑦,𝑧′)

]︀

max𝜋

𝜃

where (𝑧𝑆(1),...,𝑧𝑆(Ttrain)) ∼ Alg(𝜋𝜃;𝑥), s.t. |𝑧′| ≤ 𝐻𝑅. (6.1)

This design is effective because, under the problem structure targeted by RC, optimizing each iteration locally is aligned with optimizing the full reasoning trajectory. By training each step t to produce a correct

answer, we implicitly encourage the model to generate reasoning traces and summaries that contain information, such as intermediate results or partial analyses, that is useful for subsequent steps. Because training rollouts are conditioned on these summaries, the model is also explicitly trained to exploit this accumulated information when generating later responses, increasing the likelihood of future correctness. Under the assumption that successive summarization and summary-conditioned generation steps make roughly monotonic progress toward a solution, optimizing per-step correctness is sufficient to improve trajectory-level performance, despite only explicitly optimizing individual steps. We discuss limitations of this assumption and evaluate alternative training approaches in Appendix E.

Training with a summary replay buffer (off-policy reinforcement learning). The iterative decoding structure of RC naturally enables learning from off-policy summaries because summaries serve as conditioning inputs rather than optimization targets. Learning from off-policy summaries provides two benefits. First, it enables training on later reasoning turns without the need for us to generate long on-policy decoded trajectories, which is useful because summaries from later turns may qualitatively differ from earlier ones. Second, it increases the coverage of summaries the model encounters during training, which in turn increases the robustness of the policy to test time shifts in the summary distribution. In fact, improving state coverage via a replay buffer is one of the fundamental principles in off-policy RL [6], which using past summaries enables us to implement.

We therefore incorporate a summary replay buffer into training; see Figure 7. During the first training epoch, we follow the same on-policy procedure to optimize Equation 6.1, but also store all generated summaries, and their corresponding problems, in the replay buffer ℬ. From the second epoch onward, we sample problems and summaries from ℬ and condition RC rollouts on them instead of generating fresh summaries, thereby extending the maximum effective training horizon by Ttrain steps per epoch.

##### 7. Experimental Evaluation: Training with RC

The goal of our experiments is to demonstrate the effectiveness of training with RC to improve extrapolation. To this end, we evaluate our approach on several benchmarks, compare it with related methods, and also conduct several ablation experiments to isolate the effects of our different design choices.

Training details. We post-train a Qwen3-4B-Instruct-2507 model to utilize RC and refer to the trained model as RCT-4B. We set 𝐾 = 8, 𝑁summ = 2 and Ttrain = 3. We conduct training in two stages: in Stage I, we train without the summary replay buffer, focusing on optimizing early turns, including the initial turn (without any summary context). Training problems for Stage I are subsampled from the AceReason-Math dataset [3], resulting in a dataset of about 5.7k problems. For Stage II, we enable the summary replay buffer, and construct a new training set by injecting a small number of difficult problems from DAPO [46] into our Stage I dataset as part of our training curriculum. See Appendix I for details.

Benchmarks and evaluation protocols. We evaluate RCT-4B on three math reasoning benchmarks: AIME 2025, HMMT 2025 (November version), and IMO-AnswerBench [20], as well as one scientific reasoning benchmark, FrontierScience (Olympiad) [24], which contains expert-written problems in physics, chemistry, and biology. Since our training data exclusively consists of mathematical reasoning problems, FrontierScience serves to assess whether learned extrapolation behavior generalizes to an unseen domain. We evaluate AIME and HMMT using math-verify1 for equivalence checking, and follow the official LLM judge-based evaluation protocols for IMO-AnswerBench2 and FrontierScience. We

1https://github.com/huggingface/Math-Verify 2In a previous version of this paper, we reported IMO-AnswerBench results evaluated using math-verify. We subsequently

###### AIME 2025

HMMT 2025 (Nov)

IMO AnswerBench

FrontierScience

75

34

65

57

+9.6

70

32

+4.6 +10.9

54

60

+9.0 +17.1

+15.4

Accuracy

65

30

51

55

60

+26.5

28

48

50

55

+28.8

26

45

+12.7

45

+9.4

+3.4

50

+3.6

24

42

40

45

16 64 128 192 256

16 64 128 192 256

16 64 112 160 192

16 64 112 160 192

Reasoning token budget (k)

Qwen3-4B-Instruct-2507 RCT-4B

- Figure 8: Accuracy on various reasoning benchmarks as a function of token budget for Qwen3-4B-Instruct-2507 and RCT-4B with RC decoding. RCT-4B improves performance across all four reasoning benchmarks as reasoning token budget 𝐻test is increased beyond 𝐻train. This improvement is larger than that attained by using the base model.

selected the latter three benchmarks for their low contamination risk: all three were released after our training datasets and our base model, with HMMT 2025 (Nov) and FrontierScience consisting entirely of new problems and IMO-AnswerBench consisting of extensively rewritten past Olympiad problems.

Baselines and comparisons. We compare against three categories of approaches. The first consists of autoregressive decoding methods using open-source 4B models. These include Qwen3-4B-Instruct2507 (our base model), Qwen3-4B-Thinking-2507, Polaris-4B [2], a Qwen3-4B-based model trained for extrapolation by expanding the output context using YaRN [25], and a version of Qwen3-4B-Instruct-2507 trained with standard GRPO at 𝐻train = 32k, the maximum output length we could reliably use for RL due to practical constraints. The second category of comparisons consists of other iterative decoding approaches that condition directly on raw past reasoning rather than on summaries. We evaluate base and trained (see Appendix J) versions of two such methods: self-refinement and self-verification, selecting 𝐻𝑅 and 𝐻test to be the same as for our RC experiments. We also evaluate two other iterative decoding approaches inspired by recent work [1, 23] in Appendix K. The third category of comparisons consists of approaches that do not train with RC but still use RC at inference time. We compare RCT-4B against inference-only use of RC by the base model (as in Section 5.2) and by the base model post-trained with standard RL. This comparison isolates the contribution of training with RC, rather than applying it solely at test time. See Appendix F for a discussion on the computational efficiency of RC.

###### 7.1. Benchmark Results

Our results are shown in Figure 8 and Table 1. Across all benchmarks and token budgets, RCT-4B outperforms the base model using RC, with the performance gap widening as the token budget increases. This indicates that training enables more effective extrapolation rather than merely improving short-horizon performance. Notably, the model also improves on FrontierScience despite being trained exclusively on mathematics problems, suggesting that RC training develops domain-general extrapolation capabilities.

- Finding 1: RC compares favorably against other strong 4B-sized models. We compare RCT-4B against strong 4B reasoning-specialized models that utilize autoregressive decoding. While these models are explicitly trained to exploit large token budgets, RCT-4B outperforms all autoregressive approaches on the three benchmarks with lowest contamination risk: HMMT 2025 (released in Nov ’25), IMO-AnswerBench (released in Nov ’25), and FrontierScience (released in Dec ’25). In fact, we find that RCT-4B even achieves competitive results against much larger reasoning models: see Figure 11. Notably, while standard RL

discovered that this approach underestimates accuracy, as math-verify fails to parse certain valid answer formats on this benchmark, leading to correct responses being marked as incorrect. We therefore report results using the official LLM judge-based evaluation protocol instead, replacing Gemini-2.5-Pro with Gemini-3-Flash.

###### AIME 2025 HMMT 2025 (Nov) IMO-AnswerBench FrontierScience

Qwen3-4B-Instruct-2507 [16k] 46.0 39.8 40.9 23.3 Qwen3-4B-Instruct-2507 (RL, 32k) [32k] 54.8 48.3 42.1 21.5 Polaris-4B [2] [90k] 79.4 60.2 47.7 23.6 Qwen3-4B-Thinking-2507 [81k] 81.3 62.5 53.8 25.7

Self-Refine (base) 53.8 50.0 45.8 27.8 Self-Verify (base) 48.9 46.7 43.9 29.7 Self-Refine (trained) 60.4 61.3 52.2 33.5 Self-Verify (trained) 61.2 62.1 52.3 31.9

Qwen3-4B-Instruct-2507 + RC 59.4 56.7 46.3 29.5 Qwen3-4B-Instruct-2507 (RL, 32k) + RC 66.0 60.2 53.4 33.5 RCT-4B + RC (Ours) 74.9 66.3 58.0 34.1

Table 1: Evaluation results. The top section of the table reports results of various reasoning models run with standard autoregressive generation; 𝐻test given in square brackets. The middle section reports various iterative decoding baselines, run for 12 (AIME, HMMT) or 16 (IMO-AnswerBench, FrontierScience) turns (equivalent to a 192k or 256k reasoning token budget) with 𝐻𝑅 = 16k. The final section reports results for RC-based results, which were run with the same token budgets as the iterative decoding baselines. RCT-4B outperforms all baseline methods on 3 out of 4 benchmarks.

training improves upon the base model on the mathematics benchmarks, this standard RL model (1) remains substantially weaker than specialized reasoning models, and (2) achieves no gains on the out-ofdomain FrontierScience benchmark despite training on the same data as RCT-4B. This demonstrates that RC training develops more generalizable problem-solving strategies that standard RL does not.

- Finding 2: RC training yields better iterative reasoning than other iterative training methods. RCT-4B substantially outperforms all iterative decoding methods we compare against. Training models to perform self-verification or self-refinement using an approach analogous to RC training (Appendix J) leads to large improvements over their respective base models. However, these baselines remain significantly weaker than RCT-4B on mathematical reasoning tasks, highlighting the benefit of exploiting the summarization–generation asymmetry during training. Interestingly, nearly all iterative decoding methods we evaluate, including untrained variants, outperform autoregressive decoding on FrontierScience. This suggests that iterative decoding may generalize better to out-of-domain input problems compared to standard autoregressive generation for the base models we consider in our experiments.

###### 7.2. Evaluating RC on Hard Problems

Our results thus far show that extrapolation with RCT-4B yields consistent improvements on standard benchmarks. However, these gains could have arisen either from solving harder problems with additional test-time compute or from sharpening performance on problems that are already partially solvable within 𝐻𝑅. To distinguish between these effects, we evaluate models on a set of adversarially curated problems from Omni-MATH [8], following the protocol of Qu et al. [27]. These problems are selected such that the base Qwen3-4B-Instruct-2507 model fails to produce any correct solution across 𝑁 = 256 independent attempts, even when given sufficient reasoning budget. We consider such problems to be difficult because they are unlikely to be solvable simply by scaling parallel compute [5, 41, 44].

Evaluation results. We evaluate both the base model and RCT-4B on this dataset using RC decoding and report pass@𝑘 in Figure 9. While both models improve with increased reasoning token budgets, the gains for RCT-4B are substantially larger. At a budget of 256k tokens, the base model achieves a pass@16 of 20%, whereas RCT-4B reaches nearly 35%. Moreover, the performance gap between the trained and

(RCT-4B, Qwen3-4B-Instruct-2507)

Qwen3-4B-Instruct-2507

RCT-4B

35

20

15

30

Accuracy

25

15

10

| |
|---|

| |
|---|

20

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

10

| |
|---|

| |
|---|

| |
|---|

| |
|---|

15

| |
|---|

| |
|---|

| |
|---|

| |
|---|

5

10

| |
|---|

| |
|---|

| |
|---|

5

5

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

16 64 112 160 208 256

16 64 112 160 208 256 Reasoning token budget (k)

16 64 112 160 208 256

Pass@1 Pass@4 Pass@8 Pass@16

- Figure 9: Pass@𝑘 accuracy vs. token budget on a hard subset of problems sampled from [27] for Qwen3-4B-Instruct2507 and RCT-4B. The left panel shows base model performance, the middle panel shows trained model performance, and the right panel shows the performance gap between the two. RCT-4B achieves substantially higher pass@𝑘 rates than the base model across all values of 𝑘, with the performance gap increasing as reasoning token budget grows.

base models widens as the token budget increases, indicating that training improves the model’s ability to utilize RC for in-context exploration. Overall, these results demonstrate that RCT-4B can solve difficult problems that the base model cannot by effectively extrapolating its reasoning at test time. For additional results directly comparing RC against parallel compute methods (majority vote), see Appendix B.

###### 7.3. Ablation Studies

The effect of the summary replay buffer. We ablate our training procedure by comparing performance on AIME 2025 after (i) Stage I training only, (ii) Stage II training without using the summary replay buffer, and (iii) Stage II training with the replay buffer enabled; see Figure 10 (left). Stage I training alone yields substantial gains over the base model, while Stage II training provides additional improvements. These gains are modest without the replay buffer but significantly larger when it is used, particularly at higher reasoning budgets. At 16k tokens, Stage II with replay buffer improves accuracy by 2.7% over Stage I, while at 192k tokens this gap grows to 9.4%. In contrast, Stage II without replay buffer yields only a ∼4% improvement at 192k tokens, thus demonstrating the effectiveness of our replay buffer.

The effect of the number of training turns. Next, we vary the number of training turns Ttrain during Stage I, evaluating Ttrain ∈ {2,3,4}; see Figure 10 (middle). All settings improve extrapolation, with Ttrain = 3 performing best, followed by Ttrain = 2. We believe this arises from a trade-off in gradient signal allocation: when Ttrain is too large, the model receives insufficient signal on early turns (including the initial turn), while the converse is true when Ttrain is too low. These results imply that we must train the model evenly across all early turns in order for good summary-conditioned reasoning to emerge.

The importance of training with RC. Finally, we isolate the effect of RC training by comparing against a model trained with standard outcome-reward RL (GRPO). We train Qwen3-4B-Instruct-2507 with RL for the same number of steps and following the same two-stage training curriculum while increasing the training budget 𝐻train from 16k to 32k tokens (the maximum output length we could reliably use for RL due to practical constraints). We then conduct evaluation using RC decoding; see Figure 10 (right) and Table 1 (bottom) for our results. While RL training yields modest improvements over the base model, it falls far short of training with RC. This demonstrates that effective extrapolation via RC decoding does not emerge from standard RL alone and must be trained through RC’s structured, multi-turn objective.

75

65

Acc.onHMMT'25

Acc.onAIME'25

Acc.onAIME'25

65

60

70

55

60

65

50

- Ttrain = 2

- Ttrain = 3

- Ttrain = 4

I Only

Base + RC

55

60

I + II (w/o replay)

Trained + RC

45

I + II (w replay)

Std. GRPO trained + RC

40

50

16 64 112 160 192

16 64 112 160 192

16 64 112 160 192

Reasoning token budget (k)

- Figure 10: Left: Ablation study on stagewise training configurations. Performance improves with Stage II training, with the summary replay buffer providing additional gains that increase with reasoning budget. Middle: Ablation study on Ttrain values during Stage I. Ttrain = 3 yields optimal performance across budgets by balancing early-turn training with summary-conditioned reasoning exposure. Right: Comparison of training methods. RC-specific training substantially outperforms standard GRPO training, demonstrating that extrapolation requires explicit training beyond simply improving reasoning capabilities.

###### 7.4. Incorporating RC into test-time scaffolds

Our experiments thus far show that RC training enables more effective extrapolation by improving summary-conditioned generation. We can view this as a specific instance of self-guided reasoning: the model conditions on self-generated abstractions [28, 45] (in our case, summaries of prior reasoning) to guide downstream reasoning. This raises a natural question: does RC training teach a generalizable skill for using self-generated guidance, or are improvements narrowly tied to reasoning from summaries? To answer this, we evaluate how RC training transfers to other selfguidance settings by incorporating RCT-4B and RC decoding into two test-time scaffolds: Recursive Self-Aggregation (RSA) [40] and the DeepseekMath (DSM) Agent [35]. RSA generates parallel reasoning traces and iteratively aggregates them, while the DSM Agent iteratively performs self-verification and self-refinement over an initial pool of solutions; see Appendix G. In both cases, the model is used not only to generate reasoning traces from scratch, but also to reason conditioned on self-generated context produced via the scaffold.

RSA DSM Agent Base 66.3 57.5

RL 64.8 61.3 RCT-4B (no RC) 70.2 65.8

RCT-4B + RC (Ours) 75.4 74.6

Table 2: RCT-4B yields additional gains when incorporated into test-time scaffolds. These gains are higher than with the base or standard RL-trained model. Using RC decoding within the scaffold yields further gains.

Evaluation results. We report results on HMMT 2025 (Nov) in Table 2, which shows that RCT-4B leverages both scaffolds far more effectively than the base or RL-trained Qwen3-4B-Instruct-2507 models, even without RC decoding. This suggests that RC training imparts a generalizable capability: reasoning effectively from self-generated abstractions beyond summaries, including aggregated past traces (RSA) and self-generated feedback (DSM Agent). Replacing standard autoregressive decoding with RC for all solution-generation steps within these scaffolds yields additional gains, which we attribute to improved reasoning accuracy throughout the scaffolded process. Overall, these results indicate that RC training develops abstraction-conditioned reasoning abilities, highlighting an important direction for future work.

Key Takeaways: Training with RC improves extrapolation across domains

- • RCT-4B outperforms iterative decoding and autoregressive decoding baselines.
- • RC enables models to solve difficult problems that cannot be solved by scaling parallel compute.
- • RC training yields models that are better at summary- and abstraction- conditioned reasoning in general, and can therefore effectively leverage scaffolds to further scale test-time compute.

##### 8. Conclusion and Perspectives on Future Work

In this work, we demonstrate how LLMs can be trained to continually improve their reasoning across long horizons. Our method, RC, replaces autoregressive decoding with an iterative decoding algorithm that alternates between summarization and summary-conditioned reasoning, and trains models via outcomereward RL to leverage this algorithm more effectively. We show that using RC allows us to overcome a fundamental limitation of autoregressive generation and standard RL: the inability to extrapolate reasoning beyond training rollout lengths. Empirically, we demonstrate that our RC-trained model achieves substantial performance gains on challenging mathematical and scientific benchmarks by extrapolating reasoning to much longer horizons than it was trained for. Furthermore, we show that extrapolation via RC enables the model to solve difficult problems that it cannot solve within its training budget, and that the trained model can leverage existing test-time scaffolds to further scale inference compute. Ultimately, we believe that RC represents an important step toward training models that can engage in the systematic, long-horizon reasoning required to solve the world’s most difficult problems.

Future work. Future work should address the limitations of RC (see Appendix E). We identify three main directions. First, improving the training objective beyond myopic rewards: our current approach assigns rewards based on individual trace correctness, which discourages multi-turn strategies in which early turns perform exploratory, low-reward procedures that are only exploited in later turns. As we discussed in Section 5.1, we do not expect this to be a problem when each iteration of summarization still makes useful progress, but training with a non-myopic extension of RC could potentially yield significant improvements on harder problems. Second, explicitly training for summary generation: our current approach only trains the model for summary-conditioned generation. Our experiments in Section 5.2, however, show that altering summaries can significantly affect the performance of RC, suggesting that directly optimizing summary generation could also be beneficial. This direction is also closely related to prior work on training models to produce useful abstractions for guiding reasoning [28, 45] or in training aggregation mechanisms [40], with both lines of work demonstrating tangible performance gains. Third, adapting RC to tasks without final-answer based rewards: while our current work focuses on settings with closed-form, verifiable outcomes, these tasks represent only a minority of the reasoning challenges we ultimately care about. Broadening RC to operate in open-ended domains, such as proof generation, is therefore an exciting and potentially impactful direction for future research.

##### Acknowledgements

We would like to thank Matthew Yang, Haoran Li, Lewis Tunstall, Jasper Dekoninck, and the other members of the CMU AIRe lab for discussions and feedback. We would also like to thank the CMU FLAME center for providing the compute resources on the Orchard cluster that supported almost all of our big experiments. Additionally, we are grateful for the Delta AI cluster at NCSA and the TRC program of Google Cloud for additional computational resources. This work was supported by the Office of Naval Research under ONR N0014-24-2206 and a Schmidt Sciences AI2050 Early Career Fellowship. AS is supported by a JP Morgan AI PhD fellowship and YQ is supported by an Amazon AI PhD fellowship.

##### References

- [1] Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, and Siva Reddy. The markovian thinker: Architecture-agnostic linear scaling of

- reasoning, 2025. URL https://arxiv.org/abs/2510.06557.
- [2] Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp.github.io/ blog/2025/Polaris.
- [3] Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025.
- [4] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025. URL https://arxiv.org/abs/2501.17161.
- [5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.
- [6] Justin Fu, Aviral Kumar, Matthew Soh, and Sergey Levine. Diagnosing bottlenecks in deep Q-learning algorithms. arXiv preprint arXiv:1902.10250, 2019.
- [7] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars, 2025. URL https://arxiv.org/abs/2503.01307.
- [8] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omnimath: A universal olympiad level mathematic benchmark for large language models, 2024. URL https://arxiv.org/abs/2410.07985.
- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi

- Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645 (8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.
- [10] Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu, Yichi Yang, Zhijian Liu, Zhiting Hu, and Lianhui Qin. Arcmemo: Abstract reasoning composition with lifelong llm memory, 2025. URL https://arxiv.org/abs/2509.04439.
- [11] Kelly Hong, Anton Troynikov, and Jeff Huber. Context rot: How increasing input tokens impacts llm performance. Technical report, Chroma Research, July 2025. URL https://research.trychroma. com/context-rot.
- [12] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet, 2024. URL https: //arxiv.org/abs/2310.01798.
- [13] Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks,

2023. URL https://arxiv.org/abs/2303.17491.

- [14] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust. Training language models to self-correct via reinforcement learning, 2024. URL https://arxiv.org/abs/ 2409.12917.
- [15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [16] Yucheng Li, Bo Dong, Chenghua Lin, and Frank Guerin. Compressing context to enhance inference efficiency of large language models, 2023. URL https://arxiv.org/abs/2310.06201.
- [17] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023. URL https://arxiv.org/abs/2307.03172.
- [18] LM-Provers, Yuxiao Qu, Amrith Setlur, Jasper Dekoninck, Edward Beeching, Jia Li, Ian Wu, Lewis Tunstall, and Aviral Kumar. Qed-nano: Teaching a tiny model to prove hard theorems. https://huggingface.co/spaces/lm-provers/qed-nano-blogpost, 2026. Blog post.

- [19] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.
- [20] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. Towards robust mathematical reasoning, 2025. URL https://arxiv.org/ abs/2511.01846.
- [21] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.
- [22] Lovish Madaan, Aniket Didolkar, Suchin Gururangan, John Quan, Ruan Silva, Ruslan Salakhutdinov, Manzil Zaheer, Sanjeev Arora, and Anirudh Goyal. Rethinking thinking tokens: Llms as improvement operators, 2025. URL https://arxiv.org/abs/2510.01123.
- [23] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling,

2025. URL https://arxiv.org/abs/2501.19393.

- [24] OpenAI. Evaluating ai’s ability to perform scientific research tasks (frontierscience). OpenAI Research Publication, December 2025. URL https://openai.com/index/frontierscience/. Accessed: 2026-01-20.
- [25] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models, 2023. URL https://arxiv.org/abs/2309.00071.
- [26] Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve, 2024. URL https://arxiv.org/abs/2407.18219.
- [27] Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, and Aviral Kumar. Learning to reason on hard problems with privileged on-policy exploration. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025. URL https://openreview.net/forum?id=zKn6mVwPZE.
- [28] Yuxiao Qu, Anikait Singh, Yoonho Lee, Amrith Setlur, Ruslan Salakhutdinov, Chelsea Finn, and Aviral Kumar. Rlad: Training llms to discover abstractions for solving reasoning problems, 2025. URL https://arxiv.org/abs/2510.02263.
- [29] Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement fine-tuning,

2025. URL https://arxiv.org/abs/2503.07572.

- [30] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.
- [32] Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. Scaling test-time compute without verification or rl is suboptimal. arXiv preprint arXiv:2502.12118, 2025.

- [33] Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz, and Aviral Kumar. e3: Learning to explore enables extrapolation of test-time compute for llms, 2025. URL https://arxiv.org/abs/2506.09026.
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.
- [35] Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning, 2025. URL https://arxiv.org/abs/2511.22570.
- [36] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [37] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.
- [38] Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Rl grokking recipe: How does rl unlock and transfer new algorithms in llms?, 2025. URL https://arxiv.org/abs/2509.21016.
- [39] Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory, 2025. URL https://arxiv.org/abs/2504. 07952.
- [40] Siddarth Venkatraman, Vineet Jain, Sarthak Mittal, Vedant Shah, Johan Obando-Ceron, Yoshua Bengio, Brian R. Bartoldson, Bhavya Kailkhura, Guillaume Lajoie, Glen Berseth, Nikolay Malkin, and Moksh Jain. Recursive self-aggregation unlocks deep thinking in large language models, 2025. URL https://arxiv.org/abs/2509.26626.
- [41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. URL https://arxiv.org/abs/2203.11171.
- [42] Colin Wei, Jason Lee, Qiang Liu, and Tengyu Ma. Regularization matters: Generalization and optimization of neural nets vs their induced kernel. 2019.
- [43] Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, WangCheng Kang, and Derek Zhiyuan Cheng. Evo-memory: Benchmarking llm agent test-time learning with self-evolving memory, 2025. URL https://arxiv.org/abs/2511.20857.
- [44] Ian Wu, Patrick Fernandes, Amanda Bertsch, Seungone Kim, Sina Pakazad, and Graham Neubig. Better instruction-following through minimum bayes risk, 2025. URL https://arxiv.org/abs/ 2410.02902.
- [45] Matthew Y. R. Yang, Hao Bai, Ian Wu, Gene Yang, Amrith Setlur, and Aviral Kumar. Int: Selfproposed interventions enable credit assignment in llm reasoning, 2026. URL https://arxiv. org/abs/2601.14209.

- [46] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.
- [47] Charlie Zhang, Graham Neubig, and Xiang Yue. On the interplay of pre-training, mid-training, and rl on reasoning language models, 2025. URL https://arxiv.org/abs/2512.07783.
- [48] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023. URL https://arxiv.org/abs/2304.11277.
- [49] Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents, 2025. URL https://arxiv.org/abs/2506.15841.

### Appendices

##### A. Additional Results vs. Larger Reasoning Models

65.9

61.5

IMO-AnswerBenchAcc.

60

57.9 58.0

53.8 53.8

52.9

51.4

50

49.2

47.7

44.5

40.9

40

30

Qwen3-4B-Instruct-2507RCT-4B(16k) Polaris-4BQwen3-80B-Next-InstructQwen3-32BQwen3-30B-A3B-Instruct-2507Qwen3-235BQwen3-4B-Thinking-2507Nemotron-3-Nano-30B-A3BRCT-4B(256k)Qwen3-30B-A3B-Thinking-2507gpt-oss-20B

Figure 11: Comparison of RCT-4B and a selection of other reasoning models on IMO-AnswerBench. Combining RC training and decoding enables our 4B model to outcompete many larger and newer models. We set inference hyperparameters (𝑡, 𝑝, 𝐻test) based on the recommended values provided on each model’s Hugging Face page.

##### B. Comparison with Majority Voting

Qwen3-4B-Instruct-2507

###### RCT-4B

| | | | | | |
|---|---|---|---|---|---|
| | | |Acc|@ Turn 16| |
| | | | | | |
| | | | | | |
| | | | |Accuracy<br><br>Maj@16| |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | |Acc|@ Turn 16| |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | |Accuracy<br><br>Maj@16| |
| | | | | | |
| | | | | | |

IMO-AnswerBenchAcc.

60

52

56

48

52

44

48

44

16 64 128 192 256

16 64 128 192 256

Seq. reasoning token budget (k)

- Figure 12: Accuracy and Maj@16 against sequential reasoning token budget for Qwen3-4B-Instruct-2507 and RCT-4B on IMO-AnswerBench. While majority voting can be used to improve RC, we find that utilizing compute to increase “depth” through RC is more effective than increasing “breadth” by taking majority vote over more parallel samples.

In Figure 12, we plot accuracy and Maj@16 against sequential reasoning token budgets using RC decoding (true reasoning token budget for Maj@16 is the sequential reasoning token budget multipled by 16). While majority voting can be used to improve RC, we find that utilizing compute to increase “depth” through RC is more effective than increasing “breadth” by taking majority vote over more parallel samples for our tested value of 𝑘: in other words, Maj@16 performance at 16k sequential reasoning token budget

gpt-oss-20B

gpt-oss-120B

IMO-AnswerBenchAcc.

64

78

76

60

74

56

72

52

70

48

68

64 128 192 256 320

64 128 192 256 320

Reasoning Tokens (k)

- Figure 13: Performance of gpt-oss models with RC decoding (no training) on IMO-AnswerBench. We use 𝐻𝑅 = 64k to adjust for the models’ larger output lengths, and set reasoning effort to “high” for generation and “medium” for summarization. Both models benefit from extrapolation through RC. We evaluate answer correctness using Gemini-3-Flash as an LLM judge.

(giving a true reasoning token budget of 256k) is significantly worse than accuracy at 256k tokens with RC. This applies both for the base Qwen3 model and for RCT-4B.

##### C. Motivating our Choices of 𝐻𝑅

In this section, we motivate the choices of 𝐻𝑅 (autoregressive decoding maximum token budget) we use throughout this work. For RC decoding, 𝐻𝑅 determines the length of individual reasoning traces within each turn. As discussed in Section 5.2, we generally choose 𝐻𝑅 to be 𝐻train. Unfortunately, the exact value of 𝐻train is typically not made public, so we must estimate it through the termination length statistics of the model: if the model generally terminates its reasoning within some length 𝐿, then we can reasonably say that 𝐿 ≈ 𝐻train.

Regardless, the general idea is that increasing the reasoning token budget beyond 𝐿 will not yield any gains in performance with autoregressive decoding, as the model will simply not generate anything longer than this. For our base Qwen3-4B-Instruct-2507 model, we set 𝐻𝑅 = 16k, which yields a very high termination rate of 99.17% (see Table 3. Our trained RCT-4B model also attains a very high termination rate of 98.75%, indicating that our training has not resulted in increased repetitiveness or undue verbosity. For our autoregressive decoding baselines, we choose the maximum token budget based on the values recommended on each model’s Hugging Face model card. As we see from Table 3, reasoning traces generated at these lengths do indeed overwhelmingly terminate successfully.

A note on input context windows: many models have a stated maximum context windows that are very large. For example, Qwen3-4B-Instruct-2507 has a stated maximum context window of 262,144 tokens. However, we note that they almost never generate reasoning traces of lengths greater than 16k tokens: see Table 3. This is likely because the models were post-trained to generate outputs of up to 16k tokens in length: the 262,144 context window is only utilized for processing long-context inputs.

###### D. RC with gpt-oss Please see results showing the efficacy of our approach on this model family in Figure 13.

|Model|𝐻𝑅 Termination Rate (%)<br><br>|
|---|---|
|Qwen3-4B-Instruct-2507 Qwen3-4B-Thinking-2507 Polaris-4B Qwen3-4B-Instruct-2507 + Std. RL @32k RCT-4B|16k 99.17<br><br>81k 100.00<br><br>90k 99.79 32k 99.17 16k 98.75<br><br>|

- Table 3: Termination rates for different models on HMMT 2025. We determine termination by the presence of the boxed{} pattern in the model output.

##### E. Limitations of RC and Directions for Future Work

While RC training yields strong empirical results, our method has several limitations that we outline in this section. We hope this discussion provides useful directions for future work.

RC training does not optimize summary generation. Our training focuses exclusively on summary-conditioned reasoning, based on the observation that this is the primary performance bottleneck while summary generation is inherently easier for pre-trained and instruction-tuned models. We validated this assumption through preliminary experiments where we assigned rewards to summary generation based on the proportion of subsequent reasoning traces (conditioned on the summary) that produced correct answers. More formally, we optimized the following objective:

| |
|---|

60

Acc.onHMMT'25

| |
|---|

55

| |
|---|

| |
|---|

Base

50

Reasoning Trn.

45

Summarization Trn.

Both Trn.

40

16 64 112 160 192

Reasoning token budget (k)

[︃

𝑟(𝑦,𝑧𝑅,𝑘(𝑡+1))]︃]︃,

[︃E𝑧(𝑡)

1 𝐾

∑︁𝐾

Figure 14: Training for summarization generation using objective E.1 hurts performance. Training this way either in isolation or in combination with the usual summary-conditioned reasoning objective negatively impacts performance.

max𝜋

E 𝑥,𝑦∼𝒟train

𝑆 ∼𝜋𝜃(·|ℐ𝑆,𝑥,𝑧𝑅(𝑡),𝑧𝑆(𝑡−1))

𝑡∼𝑈[1,Ttrain]

𝜃

𝑘=1

where 𝑧𝑅,𝑘(𝑡+1) ∼ 𝜋𝜃(·|𝑥,𝑧𝑆(𝑡)) for 𝑘 = 1,...,𝐾. (E.1)

We tested this both in isolation and in combination with our usual summary-conditioned generation objective, using the same hyperparameters as in Section 6 for Stage I training. As illustrated in Figure 14, optimizing for summary generation only (“Summarization Trn.”) hurts the efficacy of training, such that the resulting model is no better than the base model. Optimizing for both summary generation and summary-conditioned generation (“Both Trn.”) improves performance relative to the base model but hurts performance relative to the model trained only for summary-conditioned generation (“Reasoning Trn.”). These findings motivate the design of our main approach.

We attribute these results to difficulties in credit assignment for summary generation. Even when the model generates faithful, informative summaries, it receives zero reward if subsequent reasoning fails to solve the problem, which may occur simply because the problem is too difficult to solve in a single turn and not because the summary is poor. This misalignment between summary quality and reward signal makes it difficult to effectively train summarization, although we posit that doing so effectively could further improve RC performance. Addressing this likely requires alternative reward assignment schemes for summary generation, which we leave to future work.

RC training uses myopic rewards. RC is trained by optimizing Equation 6.1 using outcome-based rewards assigned independently at each iteration. That is, the reasoning trace generated at turn 𝑡 receives reward based solely on its own correctness, without explicit credit assignment to future turns 𝑡 + 1,𝑡 + 2,....

- As discussed in Section 6, this design relies on the observation that training each step to produce a correct answer implicitly encourages the model to generate reasoning and summaries that contain information useful for subsequent iterations. When each summarization and summary-conditioned generation step makes monotonic progress toward a correct solution, such myopic rewards are well aligned with trajectory-level success. In this regime, summaries serve as sufficient representations of past reasoning, and improving per-step correctness also improves the quality of information available for future steps. Indeed, under the assumption that the model can reliably extract maximal useful information from a summary when needed, optimizing per-step correctness is sufficient to optimize the long-horizon objective over multiple iterations. Moreover, by conditioning training rollouts on these summaries, the model is explicitly trained to exploit accumulated information, further increasing the likelihood of success at subsequent steps. As a result, we can optimize correctness over the full reasoning trajectory despite only explicitly optimizing individual steps.

The main limitation of our approach is that the model is not incentivized to generate reasoning that is suboptimal for the current step but valuable later. For example, the model may benefit from exploring alternative solution strategies or collecting auxiliary information that only becomes useful later in the trajectory. Learning such far-sighted reasoning behaviors may be particularly important for very difficult problems requiring extensive in-context exploration. However, designing reward schemes that effectively encourage such multi-step contributions remains challenging, and we leave this direction to future work.

Summarization-generation asymmetry is not present in all LLMs. Our analysis in Section 5.2 reveals that RC yields the most benefits when the underlying model possesses strong summarization-generation asymmetry, and that instruction-following models generally possess this asymmetry whereas highly specialized reasoning models do not. This limits the kinds of models we can apply RC to. We propose several potential solutions to this problem. The first involves warmstarting the reasoning model for summarization and summary-conditioned generation, perhaps through distillation or SFT. This approach, however, may potentially alter the reasoning behavior of the model in a detrimental way. The second solution involves using a separate model to perform summarization generation, which we previously identified as a particularly difficult task for specialized reasoning models. This approach, however, would then require us to maintain two separate models, which could pose certain practical challenges.

RC does not improve performance on all classes of reasoning problems. While our experiments show that RC decoding and training improves model performance across mathematical and scientific reasoning benchmarks, we posit that not all classes of problem classes benefit from RC. One class of such problems are search-heavy problems, where the model must iterate through a large number of possible outcomes and select the optimal choice. The main issue here is that the redundancy property no longer applies as strongly as before, as many tokens generated may be important as they document the search process and keep track of what has been tried. Summarizing such traces risks discarding important information that may reduce test-time performance on the search task.

When is RC helpful? To understand the classes of problems for which RC is most effective, it is useful to conceptualize the solution space as a graph, where nodes represent conclusions, intermediate results, or other salient states, and edges represent logical transitions between them. RC is particularly well-suited to problems whose solution graphs are clique-like: nodes tend to form loosely connected clusters that can be summarized compactly, with relatively sparse connections between clusters. Mathematical and scientific reasoning often exhibit this structure, as progress can be decomposed into distinct conceptual advances that admit concise summaries.

In such settings, RC enables effective reasoning by allowing each iteration to explore different regions of the graph, while maintaining long-term progress by summarizing and tracking information within individual clusters. In contrast, for search-heavy problems this clique structure is largely absent, and effective reasoning requires tracking a large number of individual nodes encountered during exploration. In these cases, summaries must enumerate prior states rather than abstract them, causing limited-length summaries to quickly become overwhelmed and reducing the effectiveness of iterative summarization.

In addition to scientific and mathematical reasoning, RC may also be helpful on tasks where actions yield environment feedback that is noisy and can benefit from summarization (e.g. coding with interpreter feedback). In this case, summaries may be used to keep track of environment state, as has been explored in related work [49]. Unlike these works, we focus primarily on creating abstractions of reasoning rather than environment states, but these ideas are closely related and can likely be combined.

##### F. Discussions on Computational Efficiency

###### F.1. Inference

We begin by analyzing the computational efficiency of RC decoding compared to standard long-context autoregressive generation. We examine how RC scales with reasoning length and whether extrapolating via RC is more efficient than training models to natively handle larger token budgets through autoregressive decoding.

Notation and definitions. Let 𝐶 be the input problem length, and let 𝑁 be the maximum generation length for standard autoregressive decoding. Under RC decoding, the model proceeds for T turns, generating at each turn a reasoning statement of length ≤ 𝐻𝑅 followed by a summary of length ≤ 𝐻𝑆, with 𝐻𝑅 ≫ 𝐻𝑆. Our analysis focuses on decoder-only transformers with KV-cached decoding, where for long contexts, attention computation dominates and scales linearly with current context length.

Autoregressive

RC, HR = 8k

10000

RC, HR = 16k

Tokens/s

8000

6000

4000

4 8 16 32 64 Reasoning token budget (k)

Standard Long-Context Generation. In standard autoregressive decoding, the model generates 𝑁 tokens in a single trajectory, with the context growing from length 𝐶 to 𝐶 + 𝑁. With KV caching, the incremental cost of generating the 𝑖-th token scales linearly with the context length 𝐶 + 𝑖. Summing over all tokens, the total attention-dominated inference cost (IC) scales as

Figure 15: Plot of decoding throughput against reasoning token budget. RC decoding throughput remains constant as reasoning token budget, whereas throughput for standard autoregressive decoding decreases.

IC standard ∝

∑︁𝑁

(𝐶 + 𝑖) = 𝑁𝐶 +

𝑖=1

- 1

- 2

𝑁(𝑁 + 1) = Θ(𝑁(𝐶 + 𝑁)) (F.1)

RC Inference. In RC, each reasoning step is conditioned only on the original prompt and the current summary, rather than the full previous chain-of-thought. As a result, the effective context length within each turn is bounded by approximately 𝐶 + 𝐻𝑆 + 𝐻𝑅, and does not grow across turns. Across T turns, the total inference compute is therefore

IC RC ∝ 𝑇 · 𝐻𝑅 (𝐶 + 𝐻𝑆 + 𝐻𝑅) (F.2)

Inference Speedup. For a fixed effective reasoning budget 𝑁 = T𝐻𝑅, the inference speedup of RC is approximately

ICstandard IC RC ≈

𝐶 + 𝑇𝐻𝑅 𝐶 + 𝐻𝑆 + 𝐻𝑅 ≈ 𝑇. (F.3)

Speedup =

- as T𝐻𝑅 ≫ 𝐶 and 𝐻𝑅 ≫ 𝐻𝑆. Therefore, to reach 𝑁 = T𝐻𝑅 effective reasoning tokens, RC can be ≈ T times cheaper than autoregressive decoding in attention-dominated regimes.

Empirical study. We conduct experiments to study the computational efficiency of RC. We run RC decoding using Qwen3-4B-Instruct-2507 and standard autoregressive decoding using Qwen3-4B-Thinking-2507

- at different reasoning token budgets, logging throughput on HMMT 2025 (30 prompts) with 8 parallel

rollouts. For the RC runs, we experiment with 𝐻𝑅 ∈ {8k,16k}. We plot decoding throughput against reasoning token budget in Figure 15. This plot demonstrates that autoregressive decoding throughput rapidly decreases as token budgets increase, whereas RC decoding throughput remains constant. This is expected because RC maintains bounded context length across turns even as the effective reasoning horizon grows. RC therefore proves substantially more efficient than autoregressive decoding despite our use of a highly optimized inference engine for the latter and a naive implementation for the former: see Appendix L for implementation and hardware details.

###### F.2. Training

Standard long-context RL baseline. In both standard RL training, we perform on-policy RL (e.g., GRPO) with batch size (problems per step) = 𝐵 and samples per problem = 𝐾 (GRPO group size). Then each step generates roughly 𝐵 · 𝐾 · 𝑁 tokens, where 𝑁 is the sequence length. Since the attention cost scales with the growing context length, the forward generation compute scales as:

GenCompute standard ∝ 𝐵 · 𝐾 · 𝑁(𝐶 + 𝑁). (F.4) Including backward and optimizer computation introduces a constant multiplicative factor 𝛾, yielding

TrainCompute long ≈ 𝛾 · 𝐵 · 𝐾 · 𝑁(𝐶 + 𝑁). (F.5) When 𝑁 is large, this scales quadratically with the rollout horizon.

RC Training. RC training separates trajectory construction from policy optimization. Each training step consists of: (1) summary-trajectory generation: The model runs RC for Ttrain turns to produce a sequence of summaries; (2) policy optimization: from this trajectory, 𝑁summ summaries are sampled, and for each summary, 𝐾 reasoning rollouts of length at most 𝐻𝑅 are generated and optimized via GRPO.

The total forward generation compute per training step scales as

GenCompute RC ∝ 𝐵 · (Ttrain + 𝐾𝑁summ) · 𝐻𝑅 (𝐶 + 𝐻𝑆 + 𝐻𝑅). (F.6) Including backward and optimizer cost yields

TrainCompute RC ≈ 𝛾 · 𝐵 · (Ttrain + 𝐾𝑁summ) · 𝐻𝑅 (𝐶 + 𝐻𝑆 + 𝐻𝑅). (F.7)

Crucially, all optimized rollouts remain bounded by length 𝐻𝑅, regardless of the total effective reasoning horizon supported at inference time.

Training-Time Scaling Comparison. To reach an effective horizon 𝑁 = Ttarget 𝐻𝑅, standard long-context RL training incurs compute scaling annroximatelv as

TrainCompute standard ∝ 𝐵 · 𝐾 · T2target 𝐻𝑅2 (F.8) while RC training scales as

TrainCompute RC ∝ 𝐵 · (Ttrain + 𝐾𝑁summ) · 𝐻𝑅2 (F.9) Thus, the relative cost satisfies

TrainCompute RC TrainCompute standard ≈

Ttrain𝐾𝑁summ 𝐾 · T2target

. (F.10)

This highlights a key advantage of RC: naively increasing rollout length leads to quadratic growth in training cost, whereas RC decouples the optimized rollout length from the effective reasoning horizon. By using summaries and replay, RC enables training policies that generalize to very long reasoning horizons without incurring prohibitive quadratic costs during optimization.

- F.3. Inference KV-Cache Memory The KV cache memory footprint for autoregressive decoding scales linearly with the context length:

Memory standard ∝ 𝐶 + 𝑁, (F.11) while for RC, it is bounded by the maximum within-turn context length:

Memory RC ∝ 𝐶 + 𝐻𝑆 + 𝐻𝑅, (F.12) which is independent of T. Putting these together, RC requires ∼ 𝑇× lower KV memory at the same effective reasoning horizon:

Memorystandard MemoryRC ≈

𝐶 + 𝑇𝐻𝑅 𝐶 + 𝐻𝑆 + 𝐻𝑅 ≈ 𝑇.

##### G. Details for Test-Time Scaffold Experiments

###### G.1. Recursive Self-Aggregation

In Section 7.4, we also experiment with incorporating RC into RSA [40], a scaffold that iteratively refines solutions through sampling and aggregation. In its original form, the algorithm begins by sampling 𝑀 solutions from scratch (conditioned only on the problem). Then, in each subsequent iteration, the algorithm creates 𝑀 new solutions by randomly sampling 𝑘 candidates from the current pool of solutions (with replacement) and prompting the model to aggregate them into a single improved solution. Over the TRSA successive loops, solutions compound recursively: aggregated outputs become inputs for the next round, progressively eliminating errors and reinforcing correct solutions while maintaining a constant population of 𝑀 solutions.

We incorporate RC into RSA by replacing (1) the initial solution generation step and (2) subsequent refinement steps with RC decoding. We begin the refinement step by treating the aggregated solution as a summary that we condition on for the first step of RC refinement. For our experiments in Section 7.4, we use 𝑘 = 2, 𝑀 = 8, and TRSA = 10, and for the experiment incorporating RC decoding, we set the number of RC steps as T = 8.

###### G.2. DeepseekMath Agent

We also experiment with a test-time scaffold we call DeepseekMath Agent (DSM Agent). This is adapted from the scaffold used in Shao et al. [35] to improve the ability of LLMs to generate proofs for mathematical reasoning problems.

- At a high level, the DSM Agent implements a Generate-Verify-Refine loop that uses self-verification to iteratively improve solutions. It begins by generating an initial pool (of size 𝑛𝑔) of candidate solutions, and then verifies each solution using 𝑛𝑣 self-verification attempts per solution (assigning scores of 0.0 for major errors, 0.5 for minor issues, 1.0 for correct), with the final verification score determined by

averaging over the 𝑛𝑣 scores. In each of the subsequent refinement iterations, the algorithm selects the highest-scoring solutions and refines them using feedback from their lowest-scoring verifications. Refined solutions are added to the growing pool and re-verified, with this process repeating until either (1) a perfect score is achieved, or (2) the maximum TDSM iterations are reached. At the end, the algorithm returns the highest-scoring solution as the final answer.

We incorporate RC into DSM Agent by replacing the initial solution generation step with RC decoding, with the aim of improving the quality of the initial pool of candidates. For our experiments in Section 7.4, we use 𝑛𝑔 = 8, 𝑛𝑣 = 4, and TDSM = 6, and for the experiment incorporating RC decoding, we set the number of RC steps as T = 8.

- H. Overview of GRPO GRPO optimizes the following objective:

𝒥 (𝜃) = E𝑥,𝑦∼𝒟trainE𝑧𝑖∼𝜋𝜃(·|𝑥) [︃

1 𝐾

∑︁𝐾

𝑖=1

min [︂

𝜋𝜃(𝑧𝑖 | 𝑥) 𝜋old(𝑧𝑖 | 𝑥)

𝐴𝑖,clip(︂

𝜋𝜃(𝑧𝑖 | 𝑥) 𝜋old(𝑧𝑖 | 𝑥)

,1 − 𝜖,1 + 𝜖)︂𝐴𝑖]︂]︃. (H.1)

Here, 𝑧𝑖 denotes the 𝑖th of 𝐾 independently sampled rollouts (which taken together form a “group”), and 𝐴𝑖 denotes the GRPO advantage, which is computed directly from the rewards as 𝐴𝑖 = 𝑟𝑖−mean(𝑟)

std(𝑟) ,

with the mean and standard deviation calculated over group rewards. Some intuitions behind GRPO:

- • For a fixed input 𝑥, GRPO assigns advantages to each rollout 𝑧𝑖 relative to the other 𝐾 samples in the group, so updates depend on whether 𝑧𝑖 is better or worse than its peers rather than on absolute reward values. In the case of RC training, the 𝐾 parallel rollouts are sampled under the same prompt and summary combination, so we assign higher advantages to summary-conditioned reasoning traces that are better able to leverage the summary to attain the correct answer.
- • Normalizing advantages by the group mean and standard deviation stabilizes gradients and makes updates invariant to the overall reward scale across different inputs.
- • The clipped ratio 𝜋𝜃(𝑧𝑖 | 𝑥)/𝜋old(𝑧𝑖 | 𝑥) retains PPO’s [31] trust-region approach, preventing any single high-advantage 𝑧𝑖 from applying overly large updates.

- I. Dataset Construction Details

We construct our training datasets by following some of the principles outlined in An et al. [2]. Specifically, we sample problems in a way that ensures our dataset maintains reasonable difficulty given our model.

We begin by sampling problems from the AceReason-Math [3] dataset (∼ 50k problems) and solving

- them with Qwen3-4B-Instruct-2507, setting 𝐾 = 64. We then evaluate these solutions and assign a reward score to each problem based on the average number of correct solutions our model generates. These scores are used for weighted sampling: we discard all samples that attain a score of 0.7 or greater, and downsample problems with other reward scores to attain the “J-shaped” reward curve described in Figure 2 of An et al. [2]. This procedure yields a dataset of around 5.7k samples, which we take as our Stage I training set.

After Stage I training, our model improves and so we rebalance our training dataset such that it maintains the “J-shaped” reward curve. We reannotate our Stage I dataset with the Stage I model (with standard autoregressive decoding) and once again remove samples that attain reward scores of 0.7 of greater. We then inject ∼ 500 difficult (zero-reward) problems from the DAPO [46] dataset (as determined via annotation with the base model) and ensure that our Stage II dataset contains challenging problems.

##### J. Iterative Decoding Baseline Details

In this section, we describe in detail the self-verification and self-refinement iterative decoding baselines that we compare RC against. The purpose of these baselines is to help us separate out the impact of our summarize-generate routine from the impact of using iterative decoding. As such, these baseline methods do not utilize the summarization-generation asymmetry, and instead act directly on the reasoning trace generated by the model, as is common in iterative decoding methods and test-time scaffolds [14, 26, 35].

###### J.1. Inference

Concretely, let 𝑥 denote the input prompt and let 𝑡 ∈ N index the decoding turn. Unlike RC, these baseline methods maintain only a reasoning trace 𝑧𝑅(t) at each turn, with no separate summarization step. At each turn, the reasoning trace 𝑧𝑅(t) is generated under a fixed token budget 𝐻𝑅 (we use the same 𝐻𝑅 = 16k as in our RC experiments).

For self-refinement, decoding proceeds by alternately generating reasoning traces and prompting the model to refine them. At each turn t, we sample:

𝑧𝑅(t) ∼ 𝜋𝜃(︁· | ℐrefine,𝑥,𝑧𝑅(t−1))︁, (J.1)

where ℐrefine instructs the model to improve upon its previous reasoning trace, and 𝑧𝑅(0) is initialized as the empty string.

For self-verification, the model is prompted to first verify its previous attempt before generating a correction. At each turn t, we sample:

𝑧𝑅(t) ∼ 𝜋𝜃(︁· | ℐverify,𝑥,𝑧𝑅(t−1))︁, (J.2)

where ℐverify instructs the model to verify whether its previous reasoning is correct and, if not, to provide a corrected solution. See Figures 19 and 20 for ℐrefine and ℐverify.

After T decoding turns, the final output is given by 𝑧 := 𝑧𝑅(T) for both methods. The key distinction from RC is that these baselines condition on the full previous reasoning trace 𝑧𝑅(t−1) rather than a compressed summary. As such, the model must conditionally generate from sequences up to 2𝐻𝑅 in length.

###### J.2. Training

Training follows a similar scheme to RC training, except that we generate rollouts using our baseline iterative decoding methods instead of RC decoding. The idea here is to assess whether utilizing the summarization-generation gap enables us to achieve better performance through training, or whether our iterative training strategy on its own is sufficient to attain significant gains.

More formally, at any given point in training, we run the iterative decoding algorithm for Ttrain turns for each problem 𝑥 in a training batch. We collect the reasoning traces generated from these rollouts

𝑧𝑅 := (𝑧𝑅(1),...,𝑧𝑅(Ttrain)) and then uniformly sample 𝑁trace ≤ Ttrain unique traces per problem. We then generate 𝐾 reasoning traces conditioned on each sampled trace. We assign rewards based on correctness

and compute advantages over these 𝐾 samples. Formally, the objective can be written as:

E𝑥,𝑦∼𝒟train,𝑡∼𝑈[1,Ttrain] [︂E𝑧′∼𝜋

]︂, |𝑧′| ≤ 𝐻𝑅 where (𝑧𝑅(1),...,𝑧𝑅(Ttrain)) ∼ IterativeDecoding(𝜋𝜃;𝑥). (J.3) We adopt the same training hyperparameters as for RC training in Section 6.

𝜃(·|𝑥,𝑧𝑅(𝑡)) [︀𝑟(𝑦,𝑧′)

]︀

max𝜋

𝜃

##### K. Additional Iterative Decoding Comparisons

We evaluate two additional prompting-only iterative decoding baselines beyond the self-refinement and self-verification methods described in Appendix J. Rather than conditioning directly on the full reasoning trace, these baselines modify the trace before conditioning on them for subsequent generation.

###### K.1. Budget Forcing

###### AIME 2025 HMMT 2025 (Nov) FrontierScience

Qwen3-4B-Instruct-2507 [16k] 46.0 39.8 23.3 Qwen3-4B-Instruct-2507 + RC 59.4 56.7 29.5 RCT-4B + RC (Ours) 74.9 66.3 34.1 Budget Forcing [23] 56.7 46.7 20.1

- Table 4: Comparison of budget forcing with Qwen3-4B-Instruct-2507 against the autoregressive baseline and against RC. Budget forcing yields modest gains on mathematical reasoning benchmarks, but degrades performance on FrontierScience vs. the autoregressive baseline. RC-based methods outperform budget forcing on all benchmarks.

The first baseline we evaluate is a budget forcing approach, inspired by Muennighoff et al. [23]. In this approach, we repeatedly append “Wait, let me continue thinking” after termination to elicit additional reasoning. We use 𝐻𝑅 = 16k, and report our results in Table 4.

- Table 4 shows that budget forcing yields modest gains on mathematical reasoning benchmarks relative to autoregressive decoding but degrades FrontierScience performance. Both RC-based methods substantially outperform budget forcing across all benchmarks. Analyzing output traces reveals two failure modes. First, budget forcing rarely changes the final answer after the first few turns, with continuations quickly reducing. We hypothesize that simply appending “Wait...” is insufficient encouragement for the model to engage in meaningful in-context exploration, unlike conditioning on entire summaries of past reasoning.

Second, excessive forcing steps often produce degenerate behavior characterized by substantial repetition, especially in later turns. We attribute these issues to repeated “Wait...” prompts creating highly unnatural, out-of-distribution prefixes, especially after the model has already returned a final answer. In contrast, conditioning on summaries maintains in-distribution prefixes, yielding better performance.

###### K.2. Delethink

Our second baseline is a prompting-only version of Delethink [1]. At each iteration, Delethink generates 𝐻𝑅 tokens of reasoning conditioned on the final 𝐻chunk tokens from the previous iteration, continuing until EOS or reaching maximum iterations. This approach enables long-horizon RL training by chunking reasoning into segments, which circumvents the practical difficulties of standard long-context RL (a goal also shared by RC training: see Appendix F).

55

Acc.onHMMT'25

50

45

40

Autoregressive (16k)

35

RC

30

Delethink (8k) Delethink (4k)

25

20

15

4 36 68 100 132 164 192 Reasoning token budget (k)

We compare RC with the base Qwen3-4B-Instruct-2507 against Delethink using the same model. We evaluate Delethink with 𝐻𝑅 ∈ {4k,8k}, setting 𝐻chunk = 𝐻𝑅/2, and report our findings in Figure 16. We do not experiment with 𝐻𝑅 = 16k because the model overwhelmingly terminates its outputs at this value of 𝐻𝑅, preventing Delethink from yielding improvements at larger token budgets because the model simply will not continue generating. We find that Delethink generally improves performance as token budget increases, but plateaus near the performance of the autoregressive baseline (note that these results are consistent with findings in Aghajohari et al. [1]). RC, on the other hand, improves significantly over the autoregressive decoding baseline, even without any RC-specific training.

Figure 16: Performance as a function of reasoning token budget, comparing RC with Delethink [1]. We use Qwen3-4B-Instruct-2507 for both experiments, without any additional training. RC outperforms Delethink across all token budgets.

We attribute this discrepancy to two factors. First, Delethink terminates after generating a final answer, which limits continued reasoning. In contrast, RC explicitly encourages verifying and refining prior answers, because subsequent reasoning is driven by the produced summary from the previous iteration. We find that this process drives significant performance gains. Second, Delethink conditions generation on only the prompt and the last 𝐻chunk reasoning tokens from the previous iteration, creating out-ofdistribution prefixes for the base model. RC instead conditions generation on structured summaries of past reasoning, which, for models with strong instruction-following, is considerably more in-distribution. In particular, we hypothesize that on difficult problems, once a model has ceased to make meaningful progress within a chain-of-thought, conditioning on a short carryover from the previous iteration is often insufficient to induce reasoning that continues to make progress in the next turn. In contrast, RC partially alleviates this issue by conditioning on explicit summaries, which frequently prompt the model to restart, explore alternative strategies, or verify prior attempts. This difference may help explain why Delethink-style approaches tend to plateau, whereas RC continues to improve with additional iterations.

##### L. Hardware, Hyperparameters, and Implementation Details

Hardware. We conduct training on a single node of 8×H100 GPUs, and conduct inference on either a single node of 8×H100 GPUs or on a single node of 4×GH200 GPUs. For our inference efficiency

- Table 5: Training hyperparameters for all training experiments.

Hyperparameter Value

Learning rate 1 × 10−6 KL loss coefficient 0.001 Entropy loss coefficient 0.0 Training batch size 64 Minibatch size 32 Clip range (low) 0.2 Clip range (high) 0.28 Gamma (𝛾) 1.0 Lambda (𝜆) 1.0 Warmup schedule Constant Weight decay 0.01 Inference Temperature 1.0 Inference top-𝑝 1.0 Optimizer AdamW

experiments in Appendix F, we use 4×GH200 GPUs.

Implementation Details. We use verl [36] with FSDP [48] for training, and vLLM [15] for inference. We use a naive implementation of RC decoding for both inference and training rollout generation. At each turn, we generate reasoning traces and summaries in batches sequentially, waiting for all sequences in a batch to complete before progressing to the next turn. This is suboptimal because sequences finish at different times (due to varying lengths and early termination), causing idle GPUs to wait for the slowest sequence in each batch. A more efficient implementation would perform asynchronous decoding across turns, allowing faster sequences to progress immediately without waiting for slower ones. Despite our use of a suboptimal implementation of RC, we still empirically observe significant efficiency benefits at longer reasoning lengths (Appendix F).

Inference Hyperparameters. We perform inference with 𝑡 = 1.0 and 𝑝 = 1.0 for RCT-4B. For Qwen34B-Instruct-2507, we use 𝑡 = 0.7 and 𝑝 = 0.8. For Qwen3-4B-Thinking-2507, we use 𝑡 = 0.6 and 𝑝 = 0.95.

##### M. Full Example Example Problem

|Let ABCD and CEFG be squares such that C lies on segment DG and E lies on segment BC. Let O be the circumcenter of triangle AEG. Given that A, D, and O are collinear and AB = 1, compute FG.<br><br>|
|---|

###### Example Summary

In this solution, the goal is to compute the side length FG of square CEFG given that ABCD and CEFG are

squares with C on segment DG, E on segment BC, A, D, and O (the circumcenter of triangle AEG) are collinear, and AB = 1. The solution begins by placing square ABCD on the coordinate plane with A at (0,0), B at (1,0), C at (1,1), and D at (0,1). Point E lies on segment BC, so it is parameterized as (1, e) for e \in [0,1]. The square CEFG is then constructed with side CE, and both clockwise and counterclockwise rotations are considered to determine the position of point G. In the counterclockwise

case, G is found at (2-e, 2e-1); in the clockwise case, G is at (e,1). The condition that C lies on segment DG is then tested: for both orientations, the only value of e that satisfies this (with C

strictly on the segment) is e=1, which results in E = C and a degenerate square. This degeneracy is unacceptable, so the solution attempts to resolve the issue by using the collinearity of A, D, and the circumcenter O of triangle AEG as a primary constraint instead of the segment condition.

In the counterclockwise orientation, the circumcenter of triangle A(0,0), E(1,e), and G(2-e,2e-1) is computed using perpendicular bisectors, and the condition that this circumcenter lies on line AD (the line x=0) is imposed. Solving the resulting equation leads to a cubic in e: 3e^3 -7e^2 +3e +1 = 0, which has roots at e=1 and e = \frac{2 \pm \sqrt{7}}{3}, only one of which lies in [0,1], namely e=1. Similarly, in the clockwise case, the same condition leads to a different cubic that also only allows e =1 as a valid solution. Despite multiple attempts to find a non-degenerate configuration, all solutions

reduce to e=1, which implies that side length FG = |CE| = 0, a degenerate case. Given that this configuration fails to yield a nontrivial solution and that all attempts to satisfy the geometric constraints result in degeneracy, the only possible conclusion is that the problem likely intends a nondegenerate solution with a specific symmetric value. Based on known similar problems in geometry involving squares, collinearity, and circumcenters, the only plausible and symmetric value for FG when AB = 1 is \sqrt{2}. Although a complete non-degenerate derivation is not fully realized in this solution, the consistency with known problem structures and the absence of any non-degenerate solution under the given constraints lead to this conclusion. Therefore, the value of FG is taken as \sqrt{2}.

###### Example Summary-Conditioned Generation

We are given a geometric configuration involving two squares: ABCD and CEFG, with certain constraints:

- - ABCD and CEFG are squares.

- - C lies on segment DG.

- - E lies on segment BC.

- - O is the circumcenter of triangle AEG.

- - A, D, and O are collinear.

- - AB = 1, and we are to compute FG. Our goal is to find FG, the side length of square CEFG.

- ---

- ### Step 1: Place the configuration in the coordinate plane Let us set up coordinates to model the situation. Let square ABCD have side length 1. Place point A at (0,0), B at (1,0), C at (1,1), and D at (0,1). So, square ABCD goes counterclockwise: A \to B \to C \to D. Now, square CEFG is built on segment CE, with E on segment BC. Since BC is vertical from (1,0) to (1,1), we can parameterize point E as:

- E = (1, e), \quad \text{where } 0 \leq e \leq 1

Now we need to determine the location of G, the vertex of square CEFG. We have two possible squares on segment CE: one going **clockwise** and one **counterclockwise**. So we

must consider both orientations. We’ll keep both possibilities, and determine which one satisfies the conditions:

- 1. C lies on segment DG

- 2. A, D, and the circumcenter O of triangle AEG are collinear We already know from the previous attempt that the only solution found leads to degeneracy at e = 1, so we must find a **non-degenerate**configuration.

---

- ### Step 2: Two possible orientations of square CEFG We now define G in both cases using rotation. We are going from C to E, and forming a square CEFG, meaning the vertices go in order C \to E \to F \to

G \to C. So the side CE is one side of the square. We need to rotate vector \vec{CE} by 90^\circ to get from E to F, and then from F to G, etc. But since we need point G, we can compute G directly as: \vec{CG} = \text{rotation of } \vec{CE} \text{ by } \pm 90^\circ

Let’s compute vectors. We have: C = (1,1), \quad E = (1,e) \Rightarrow \vec{CE} = (1 - 1, e - 1) = (0, e - 1) = (0, -(1 - e))

We rotate this vector by 90^\circ clockwise and counterclockwise to get the direction from C to G ( since in square CEFG, from C, we go to E, then from E to F, etc., but to get to G, note that G is opposite to C if CEFG is labeled sequentially). Actually, more carefully: in square CEFG, vertices in order: C \to E \to F \to G \to C. So side CE,

- then EF, then FG, then GC. So the vector from E to F is \vec{CE} rotated 90^\circ, then from F to G is that rotated 90^\circ again,

etc. Alternatively, we can write point G in terms of rotation of C or E. Actually, from C, we go to E, then turn 90^\circ left or right to get to F, etc. But here’s a trick: in square CEFG, from C to E is the vector \vec{v} = \vec{CE}, then from E to F is a 90^\circ rotation of \vec{CE}. Then F = E + R_{90^\circ}(\vec{CE}). Then G = F + R_{90^\circ}(R_{90^\

circ}(\vec{CE})) = F + R_{180^\circ}(\vec{CE}) = F - \vec{CE}. But since G = C + R_{90^\circ}(\vec{CE}),

let’s verify. Wait: in square C \to E \to F \to G \to C, vector \vec{EC} is from E to C. But better: We can go: from C to E: vector \vec{v} = \vec{CE} = (0, e - 1) Then, vector from E to F is obtained by rotating \vec{CE} by 90^\circ. Depending on orientation. Let’s define:

- - Rotation by 90^\circ **counterclockwise**: (x,y) \mapsto (-y,x)

- - Rotation by 90^\circ **clockwise**: (x,y) \mapsto (y, -x) Now, from E to F, we rotate \vec{CE} by 90^\circ. So:

- - Case 1: **counterclockwise** square \vec{EF} = R_{90^\circ}(\vec{CE}) = R_{90^\circ}(0, e - 1) = (-(e-1), 0) = (1 - e, 0)

So,

- F = E + (1 - e, 0) = (1, e) + (1 - e, 0) = (2 - e, e) Then, \vec{FG} = R_{90^\circ}(\vec{EF}) = R_{90^\circ}(1 - e, 0) = (0, -(1 - e)) = (0, e - 1) So,

- G = F + (0, e - 1) = (2 - e, e + e - 1) = (2 - e, 2e - 1) Alternatively, since G = C + \vec{CG}, and \vec{CG} = rotation of \vec{CE} by 90^\circ? Wait, in counterclockwise direction from C to E, then rotating \vec{CE} by 90^\circ **counterclockwise** around E, but earlier we just computed G directly. So yes, in counterclockwise square: G = (2 - e, 2e - 1)

- - Case 2: **clockwise** square Rotate \vec{CE} = (0, e - 1) by 90^\circ **clockwise**: R_{90^\circ}(\vec{CE}) = (e - 1, 0) So \vec{EF} = (e - 1, 0)

- Then F = E + (e - 1, 0) = (1 + e - 1, e) = (e, e) Then \vec{FG} = R_{90^\circ}(\text{EF}) = R_{90^\circ}(e - 1, 0) = (0, -(e - 1)) = (0, 1 - e) So G = F + (0, 1 - e) = (e, e + 1 - e) = (e, 1) So in clockwise orientation, G = (e, 1) Thus, summary: | Orientation | G | |------------|-----| | Counterclockwise | (2 - e, 2e - 1) | | Clockwise | (e, 1) |

- ---

- ### Step 3: Apply the condition that C lies on segment DG We now use the condition that **C = (1,1) lies on segment DG**. We’ll analyze both cases.

---

- #### Case 1: Counterclockwise square, G = (2 - e, 2e - 1) We have D = (0,1), G = (2 - e, 2e - 1) So segment DG connects (0,1) to (2 - e, 2e - 1) We want (1,1) to lie on this segment. Let’s write parametric equations. Let a point P(t) = (1 - t)D + tG, t \in [0,1] So:

- x = (1 - t)(0) + t(2 - e) = t(2 - e)

- y = (1 - t)(1) + t(2e - 1) = 1 - t + t(2e - 1) = 1 + t(2e - 2) = 1 + 2t(e - 1)

We want this point to be (1,1) So set:

- x = t(2 - e) = 1 \quad \text{(1)}

- y = 1 + 2t(e - 1) = 1 \quad \text{(2)} From (2): 2t(e - 1) = 0 \Rightarrow either t=0 or e=1

- - If t=0: then from (1): 0 = 1 -> contradiction

- - So e = 1 Now, then from (1): t(2 - 1) = t(1) = 1 \Rightarrow t = 1 So e = 1, t = 1. So C is at the endpoint G? Wait, when e = 1, E = (1,1) = C, so CE is degenerate.

Then G = (2 - 1, 2 \cdot 1 - 1) = (1,1), so G = C Thus C lies on segment DG only at endpoint G, so C = G, which is endpoint. So the point C lies **on** segment DG (at endpoint), but the problem says ’C lies on segment DG’, which

often allows endpoints, but a square CEFG with E=C and G=C is degenerate - all points coincide. So we discard e=1. No valid e < 1. So **no non-degenerate solution in counterclockwise orientation**.

---

- #### Case 2: Clockwise square, G = (e, 1) Now D = (0,1), G = (e,1) Both points have y = 1, so segment DG is horizontal on line y=1 We want C = (1,1) to lie on segment DG Since DG runs from (0,1) to (e,1), it lies on y=1, and x from 0 to e For (1,1) to lie on this segment, we need 1 \in [0,e] -> e \ge 1 But E lies on segment BC, which goes from B=(1,0) to C=(1,1), so e \in [0,1] Thus, e \ge 1 and e \le 1 -> e = 1 Again, only degenerate solution. So both orientations give degenerate solutions unless e=1 But this contradicts the problem, which implies a **non-degenerate** solution. So there’s a **misunderstanding** of the configuration.

--Wait: is the square CEFG drawn outward? Or can it be on the other side? Wait: in both cases, C lies on segment DG, but maybe DG is a different segment? Wait: point D is at (0,1), so DG goes from D through G, so if G is at (e,1), then DG is from (0,1) to ( e,1), which only reaches x=e. For C=(1,1) to lie on this segment, we need e \ge 1, which is not allowed.

But in **counterclockwise** case, G = (2 - e, 2e - 1) Is it possible that in that case, DG contains C=(1,1)? We only computed when C is exactly on segment: we found only e=1 But maybe **C lies on segment DG** does **not** require e=1?

Wait - we made an algebra mistake. We had: Parametrize segment from D=(0,1) to G=(2 - e, 2e -1) We want point C=(1,1) to lie **on the segment** - not just on the ray - within the segment. So solve: x(t) = (1 - t)(0) + t(2 - e) = t(2 - e) = 1 So t = \frac{1}{2 - e} Then y(t) = (1 - t)(1) + t(2e - 1) Set y(t) = 1 So:

- (1 - t)(1) + t(2e - 1) = 1 1 - t + t(2e - 1) = 1

- - t + t(2e - 1) = 0 t( -1 + 2e - 1 ) = t(2e - 2) = 0 So again, either t=0 or 2e - 2 = 0 \Rightarrow e=1 Thus, only solution: e=1 So again - degenerated. Thus, in both configurations, C lies on DG **only when e=1**. So, **unless our assumption about the direction of square CEFG is wrong**, we have no valid solution. But the problem says "**C lies on segment DG**", and we are to assume e < 1 So what’s wrong? Ah - perhaps the square CEFG is not built externally from CE, but in another way? Wait - maybe square CEFG is built **on side CE**, but using point C and E, and then extending perpendicularly - but maybe the direction from C to G? Wait - let’s try a different labeling. Maybe CEFG means that from C to E, then E to F, F to G, G to C. So the vertex G is such that \vec{CG} is obtained by rotating \vec{CE} by 90^\circ, or -90^\circ, depending on direction. Wait - in a square CEFG, after C \to E, the turn from E to F is 90^\circ, so the direction CF? No - we can get G as C + \text{rotation}(\vec{CE}, 90^\circ)? Wait: in vector geometry, for a square CEFG, the point G such that C \to E \to F \to G \to C, then vector \vec{EG} is a 90^\circ rotation of \vec{EC}? Let’s derive a formula. Note: In square CEFG, vector \vec{EF} = R_{90^\circ}(\vec{CE}) and \vec{FG} = R_{90^\circ}(\vec{EF}) = R_{180^\circ}(\vec{CE}) = -\vec{CE} So:

- - From C to E: \vec{v} = \vec{CE}

- - E to F: R( \vec{v} )

- - F to G: - \vec{v}

- - G to C: -R(\vec{v}) So G = F + (-\vec{v}) = (E + R(\vec{v})) - \vec{v} So: G = E + R(\vec{v}) - \vec{v}

Let me test this. We have \vec{v} = \vec{CE} = (0, e - 1) Let R_{90^\circ} be **counterclockwise**: (x,y) \mapsto (-y, x) Then R(\vec{v}) = R(0, e-1) = (-(e-1), 0) = (1 - e, 0) Then G = E + (1 - e, 0) - (0, e - 1) = (1, e) + (1 - e, 0) - (0, e - 1)

= (1 + 1 - e, e) - (0, e - 1) = (2 - e, e) - (0, e - 1) = (2 - e, e - (e - 1)) = (2 - e, 1) Wait - this is different from earlier! So in this direction, G = (2 - e, 1) Similarly, for clockwise rotation: R_90^circ(\vec{v}) = (e - 1, 0) - because clockwise of (0, e-1) is (e-1, 0) Then G = E + (e - 1, 0) - (0, e - 1) = (1, e) + (e - 1, 0) - (0, e - 1) = (1 + e - 1, e) - (0, e - 1) =

(e, e) - (0, e - 1) = (e, e - (e - 1)) = (e, 1) Wait - this is the same as before!

So in counterclockwise case, G = (2 - e, 1), not (2 - e, 2e - 1) That was a mistake in previous work.So let’s correct that.

--### Revised: Correct rotation to get point G Let’s use: G = E + R(\vec{CE}) - \vec{CE}, where R(\vec{v}) is a 90^\circ rotation.

- - Case 1: Counterclockwise rotation \vec{CE} = (0, e - 1) R_{90^\circ}(\vec{CE}) = (-(e - 1), 0) = (1 - e, 0) Then: G = E + (1 - e, 0) - (0, e - 1) = (1, e) + (1 - e, 0) - (0, e - 1)

- = (1 + 1 - e, e) + (0, -e + 1)

- = (2 - e, e - e + 1) = (2 - e, 1) So, G = (2 - e, 1)

- - Case 2: Clockwise rotation \vec{CE} = (0, e - 1) R_{90^\circ}(\text{clockwise}) (0, e - 1) = (e - 1, 0) Then: G = E + (e - 1, 0) - (0, e - 1) = (1, e) + (e - 1, 0) - (0, e - 1) = (1 + e - 1, e) + (0, -e + 1) = (e,

e - e + 1) = (e, 1) So again, same as before. Thus corrected: | Orientation | G | |------------|-----| | Counterclockwise | (2 - e, 1) | | Clockwise | (e, 1) | Now check the condition that **C = (1,1) lies on segment DG**. Recall D = (0,1), C = (1,1), both on line y=1

--#### Case 1: Counterclockwise square - G = (2 - e, 1) Now D = (0,1), G = (2 - e, 1), both on horizontal line y=1 Segment DG is from x=0 to x=2 - e We need (1,1) to lie on segment DG, i.e., 0 \le 1 \le 2 - e So 1 \le 2 - e \Rightarrow e \le 1 Which is always true since e \in [0,1] And 1 \ge 0 - always true. So as long as 2 - e \ge 1, i.e., e \le 1, C lies **between** D and G? Wait: (1,1) lies between D=(0,1) and G=(2 - e, 1) if 1 lies between 0 and 2 - e So yes, when 2 - e \ge 1 -> e \le 1, which is always true. But for the point to be **on segment**, we need 0 \le 1 \le 2 - e So 1 \le 2 - e \Rightarrow e \le 1 - always true for allowed e And 1 \ge 0 - trivial So for **all** e \in [0,1], C = (1,1) lies on **segment DG**? Wait - only if 2 - e \ge 1, which is e \le 1, true, and 1 is in [0, 2-e] Yes. So every e \in [0,1] satisfies C \in DG But is this actual? For example: e = 0, E = (1,0) = B, G = (2 - 0, 1) = (2,1) Then DG: D=(0,1) to G=(2,1) - so yes, C=(1,1) is midpoint.

- For e = 1, E = C = (1,1), G = (1,1) - so G = C, and DG from (0,1) to (1,1) - so C is endpoint - still on segment. So **in counterclockwise case**, for **all** e \in [0,1], point C lies on segment DG That **solves** the earlier issue - the constraint is **satisfied for all e** in this orientation. Now, in clockwise case: G = (e, 1) So D = (0,1), G = (e,1), so segment from x=0 to x=e We want C = (1,1) to lie on this - requires e \ge 1, again only when e=1

So only degenerate. Thus, **only valid configuration is counterclockwise**, with G = (2 - e, 1), e \in (0,1)

---

- ### Step 4: Apply the collinearity: A, D, and O are collinear, where O is circumcenter of triangle AEG We have:

- - A = (0,0)

- - E = (1,e)

- - G = (2 - e, 1) We want circumcenter O of triangle AEG to lie on line AD Now, AD is from A=(0,0) to D=(0,1), so it’s the **vertical line x = 0** So, the circumcenter O of triangle A,E,G lies on x=0 We now compute the circumcenter of triangle with vertices:

- - A(0,0)

- - E(1,e)

- - G(2 - e, 1) The circumcenter is the intersection of **perpendicular bisectors**

- ---

- #### Step 4.1: Find perpendicular bisector of AE

- - Midpoint of AE: M_{AE} = \left( \frac{0+1}{2}, \frac{0+e}{2} \right) = \left( \frac{1}{2}, \frac{e}{2} \right)

- - Direction vector of AE: (1,e)

- - So perpendicular direction: (-e, 1) or (e, -1) - dot product 1(-e) + e(1) = -e + e = 0 So perpendicular bisector has slope -\frac{1}{\text{slope of } AE} = -\frac{1}{e} (if e \ne 0) Slope of AE: \frac{e - 0}{1 - 0} = e , so perp slope is -\frac{1}{e} So perp bisector: passes through (\frac{1}{2}, \frac{e}{2}), slope -\frac{1}{e} Equation: y - \frac{e}{2} = -\frac{1}{e} \left(x - \frac{1}{2} \right)

- ---

- #### Step 4.2: Perpendicular bisector of AG

- - A = (0,0), G = (2 - e, 1)

- - Midpoint M_{AG} = \left( \frac{0 + 2 - e}{2}, \frac{0 + 1}{2} \right) = \left( \frac{2 - e}{2}, \frac {1}{2} \right)

- - Direction vector: (2 - e, 1)

- - Perpendicular direction: (-1, 2 - e) or (1, -(2 - e)), since dot product: (2 - e)(-1) + (1)(2 - e) = (2 - e) + (2 - e) = 0

Take direction ( -1, 2 - e ) Slope of AG: \frac{1 - 0}{2 - e - 0} = \frac{1}{2 - e} So perp slope: - (2 - e) Thus, equation: y - \frac{1}{2} = - (2 - e) \left( x - \frac{2 - e}{2} \right)

--Now, the circumcenter O lies at intersection of these two perpendicular bisectors, and we require x = 0 So plug x = 0 into first equation and solve for e, using the two equations.

---

**Step 4.3: Plug x=0 into equation of perp bisector of AE** Equation: y - \frac{e}{2} = -\frac{1}{e} \left( 0 - \frac{1}{2} \right) = -\frac{1}{e} \left( -\frac{1}{2} \right ) = \frac{1}{2e}

So: y = \frac{e}{2} + \frac{1}{2e}

So the circumcenter has coordinates O = (0, \frac{e}{2} + \frac{1}{2e})

Now, we must also have this point satisfy the perp bisector of AG So plug x=0, y = \frac{e}{2} + \frac{1}{2e} into the equation of the perp bisector of AG: Equation: y - \frac{1}{2} = - (2 - e) \left( x - \frac{2 - e}{2} \right)

Set x = 0: Left side: y - 1/2 = \left( \frac{e}{2} + \frac{1}{2e} \right) - \frac{1}{2} = \frac{e}{2} - \frac {1}{2} + \frac{1}{2e} Right side: - (2 - e) \left( 0 - \frac{2 - e}{2} \right) = - (2 - e) \left( -\frac{2 - e}{2} \right) =

(2 - e) \cdot \frac{2 - e}{2} = \frac{(2 - e)^2}{2} Set both equal: \frac{e}{2} - \frac{1}{2} + \frac{1}{2e} = \frac{(2 - e)^2}{2} Multiply both sides by 2: e - 1 + \frac{1}{e} = (2 - e)^2 Compute right-hand side:

- (2 - e)^2 = 4 - 4e + e^2 So: e - 1 + \frac{1}{e} = 4 - 4e + e^2

Bring all terms to one side: e - 1 + \frac{1}{e} - 4 + 4e - e^2 = 0

Simplify: (5e - 5) - e^2 + \frac{1}{e} = 0

Multiply both sides by e to eliminate denominator: e(5e - 5) - e^3 + 1 = 0 5e^2 - 5e - e^3 + 1 = 0

- - e^3 + 5e^2 - 5e + 1 = 0 Multiply by -1: e^3 - 5e^2 + 5e - 1 = 0 We now solve: e^3 - 5e^2 + 5e - 1 = 0 Try rational roots: candidates +/-1

- - e=1: 1 - 5 + 5 - 1 = 0 -> **yes!** Factor out (e - 1) Use polynomial division or synthetic division: Divide e^3 - 5e^2 + 5e - 1 by e - 1

- - Coefficients: 1 | -5 | 5 | -1

- - Bring down: 1

- - Multiply: 1 \cdot 1 = 1, add to -5 -> -4

- - -4 \cdot 1 = -4, add to 5 -> 1

- - 1 \cdot 1 = 1, add to -1 -> 0 So quotient: e^2 - 4e + 1 Thus: (e - 1)(e^2 - 4e + 1) = 0 Solutions:

- - e = 1

- - e = \frac{4 \pm \sqrt{16 - 4}}{2} = \frac{4 \pm \sqrt{12}}{2} = \frac{4 \pm 2\sqrt{3}}{2} = 2 \pm \

sqrt{3} Now 2 + \sqrt{3} \approx 2 + 1.732 = 3.732 > 1 2 - \sqrt{3} \approx 2 - 1.732 = 0.268 \in (0,1) So only two valid solutions in [0,1]: e = 1, e = 2 - \sqrt{3} Now e = 1 gives degenerate square: E = C, and G = (2 - 1,1) = (1,1), so square collapses. We discard it. So valid solution: e = 2 - \sqrt{3} We now compute FG, which is the side length of square CEFG Now, CEFG has side CE C = (1,1), E = (1,e) So distance:

CE = |1 - e| = 1 - e \quad (\text{since } e < 1) We have e = 2 - \sqrt{3} So: 1 - e = 1 - (2 - \sqrt{3}) = -1 + \sqrt{3} = \sqrt{3} - 1 Thus: FG = CE = \sqrt{3} - 1 Wait - but earlier guess was sqrt2 - this is different. Is that correct? Wait: confirm the side length: yes, FG = CE, since CEFG is a square. We have e = 2 - \sqrt{3} \approx 0.2679, 1 - e \approx 0.7321, and \sqrt{3} - 1 \approx 1.732 - 1 = 0.732 - yes. So FG = \sqrt{3} - 1 But wait, is this the required answer? Let me double-check all steps. ### Verification: We assumed counterclockwise square with G = (2 - e, 1) Validated that C = (1,1) on segment DG (yes) Computed circumcenter: from perp bisector of AE, got x=0 at y = \frac{e}{2} + \frac{1}{2e} Then required it to satisfy equation from perp bisector of AG, derived cubic: e^3 - 5e^2 + 5e - 1 = 0 Roots: e=1, e = 2 \pm \sqrt{3} Only 2 - \sqrt{3} \in (0,1) Then CE = 1 - e = 1 - (2 - \sqrt{3}) = \sqrt{3} - 1 So FG = \sqrt{3} - 1 But wait - is this the simplest form? Yes. Now, is this value consistent? Check if the circumcenter is well-defined.

- For e = 2 - \sqrt{3}, all points are distinct: A, E, G not collinear? Check: three points A(0,0), E(1,e), G(2-e,1) They form a triangle? Yes, since not collinear. And O is at x=0, meaning on line AD - as required. So all conditions are satisfied.

--Earlier summary said the only plausible symmetric value is sqrt2, but clearly that’s not the case - our derivation gives a clean algebraic solution.So we reject that guess. Thus, correct answer is \boxed{\sqrt{3} - 1} Wait - is that possible? Let me compute approximate values:

- - AB = 1

- - CE = \sqrt{3} - 1 \approx 0.732

- - So FG = \sqrt{3} - 1 Yes. But the problem says "compute FG", and we have a clean expression. Alternatively, could we have expressed in another form?

Alternatively, is there symmetry? Note that in the expression: e = 2 - \sqrt{3} \Rightarrow 1 - e = \sqrt{3} - 1

Yes. So final answer: \boxed{\sqrt{3} - 1} But let me just check - perhaps the square is measured differently? Wait - the side of square CEFG is CE, which is distance between C and E. C = (1,1), E = (1,e), so vertical segment of length 1 - e Yes. So side length FG = CE = 1 - e = \sqrt{3} - 1 All conditions are satisfied. We tested the configuration: counterclockwise square, C on DG (since DG goes from D=(0,1) to G=(2-e,1),

with 2-e = 2 - (2 - \sqrt{3}) = \sqrt{3} \approx 1.732 > 1, so C=(1,1) is strictly between D and G yes. E=(1,e) with e=2-\sqrt{3} \approx 0.267, so E lies on BC, from B=(1,0) to C=(1,1) - yes. Circumcenter lies on AD - we derived it. So all constraints satisfied.

--### Conclusion The previous attempt failed because:

- - It incorrectly assumed the square orientation and coordinates.

- - It used the wrong location for point G.

- - It minimized over degenerate cases. We corrected the configuration by properly applying rotation to get point G, and verified that only the counterclockwise orientation satisfies all constraints. We solved the collinearity condition and obtained a valid non-degenerate solution.}*)

Final answer: \boxed{\sqrt{3} - 1}

##### N. RC Prompts

You are given a maths problem. You may also be given a summary of a previous attempt to solve it. This previous attempt may or may not be correct.

### PROBLEM {problem}

### SUMMARY OF PREVIOUS ATTEMPT {curr_summary}

### INSTRUCTIONS If no summary of a previous attempt is provided, solve the problem from scratch.

If a summary of a previous attempt is provided, your task is to improve upon this attempt. You should rely on this summary to guide your thinking. Some examples of strategies you could use include:

- - Verifying the previous solution.
- - Proving the result in a different way.
- - Finding alternative problem-solving strategies.
- - Continuing from where the previous solution left off, assuming that the previous solution is incomplete. Reason step-by-step and return your final answer in \\boxed{{}}.

###### Figure 17: Summary-conditioned reasoning prompt ℐ𝑅.

You are given a maths problem and a candidate solution to it. You may also be given a summary of a previous candidate solution to the problem. If this is provided, you may assume that the current candidate solution was generated conditioned on the summary of the previous candidate solution. Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:

- - It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- - It should summarize the current candidate solution in light of any previous summaries, if provided. We should be able to understand the relationship between the previous solution and the current solution by reading the summary. Make sure any important information contained in the existing summary is retained in the new one.
- - It should be no more than two paragraph long and written in paragraph form, without headers or subheaders.
- - It should be written in the first person, as if though it is being written by the person solving the problem.
- - The candidate solution may not be complete. In this case, the summary should still attempt to summarize the partial solution.

IMPORTANT: Do not under any circumstances add any additional reasoning not contained in the latest reasoning step. Your task is only to summarize what is given to you.

### PROBLEM {problem}

### EXISTING SUMMARY {existing_summary}

### LATEST CANDIDATE SOLUTION {reasoning}

###### Figure 18: Summarization prompt ℐ𝑆.

You are given a maths problem. You may also be given a previous attempt to solve it. This previous attempt may or may not be correct. ### PROBLEM {problem} ### PREVIOUS ATTEMPT {prev} ### INSTRUCTIONS If no previous attempt is provided, solve the problem from scratch. If a previous attempt is provided, start by providing feedback on this solution. Then refine the solution based on the feedback. Reason step-by-step and return your final answer in \\boxed{{}}.

###### Figure 19: Self-refinement ℐrefine prompt.

You are given a maths problem. You may also be given a previous attempt to solve it. This previous attempt may or may not be correct. ### PROBLEM {problem} ### PREVIOUS ATTEMPT {prev} ### INSTRUCTIONS If no previous attempt is provided, solve the problem from scratch. Review the previous attempt and identify possible mistakes. Afterwards, correct those mistakes and return the corrected answer. Reason step-by-step and return your final answer in \\boxed{{}}.

###### Figure 20: Self-verification ℐverify prompt.

Answer Only

- - You should extract only the final, closed-form answer provided in the current solution. This is typically contained inside \\boxed{{}}. Do not include any working or other logic.
- - If no final answer is provided, just say "No answer available". 1 - 2 Sentences
- - It should provide an overview of the approach taken in the candidate solution and in the previous summary.
- - It should be brief, no more than 1 to 2 sentences long.

- 1 Paragraph

- - It should provide a succinct overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- - It should be no more than one paragraph long and written in paragraph form, without headers or subheaders.

- 2 Paragraphs

- - It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- - It should be no more than two paragraphs long and written in paragraph form, without headers or subheaders. Multiple Paragraphs
- - It should provide a highly detailed overview of what occurred in the current candidate solution. The summary should omit any intermediate calculations (arithmetic, algebra etc.) but should describe all other steps, including failed or incorrect steps.
- - It should be written in paragraph form, without headers or subheaders. It is expected to be multiple paragraphs long and highly detailed.

###### Figure 21: Prompts from the summary length experiments in Section 5.2. The instructions are inserted into the

summarization prompt ℐ𝑆 in order to control the level of detail in the resulting summaries. The default level of detail is “2 paragraphs”.

You are given a maths problem, a summary of a previous attempt to solve the problem, and the reasoning employed in a new attempt to solve it. You are tasked with assessing how the problem-solving strategy of the new attempt is influenced by the previous attempt.

We identify these possible influences:

- A. Verification: The new solution attempts to directly verify the previous solution and its logic.
- B. Exploration: The new solution, informed by the previous solution, deliberately explores a different strategy to solve the problem.
- C. Refinement: The new solution explicitly acknowledges the previous solution and walks through the same problem-solving steps in order to refine it and build further confidence in the answer.
- D. None: The new solution does not directly depend on or reference the previous solution, and appears to be an independent attempt at solving the problem.

More than one influence may be present in the new attempt - select all the apply (A - D). D should only be selected on its own. Provide an explanation before returning your final assessment (A-D) in \\boxed{{}} (e.g. \\boxed{{B}}). If multiple influences are present, separate them with commans (e.g. \\boxed{{A, C}}).

### PROBLEM {problem}

### SUMMARY OF PREVIOUS ATTEMPT {summary}

### CURRENT ATTEMPT {reasoning}

###### Figure 22: Annotation prompt used by Qwen3-80B-Next-Instruct in Section 5.3.

