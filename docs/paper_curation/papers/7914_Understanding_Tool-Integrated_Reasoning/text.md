## arXiv:2508.19201v1[cs.LG]26Aug2025

[Figure 1]

# Understanding Tool-Integrated Reasoning

Heng Lin1,2 and Zhongwen Xu1 1Tencent, 2Tsinghua University

Abstract: We study why Tool-Integrated Reasoning (TIR) makes Large Language Models (LLMs) more capable. While LLMs integrated with tools like Python code interpreters show great promise, a principled theory explaining why this paradigm is effective has been missing. This work provides the first formal proof that TIR fundamentally expands an LLM’s capabilities. We demonstrate that tools enable a strict expansion of the model’s empirical and feasible support, breaking the capability ceiling of pure-text models by unlocking problem-solving strategies that are otherwise impossible or intractably verbose. To guide model behavior without compromising training stability and performance, we also introduce Advantage Shaping Policy Optimization (ASPO), a novel algorithm that directly modifies the advantage function to guide the policy behavior. We conduct comprehensive experiments on challenging mathematical benchmarks, leveraging a Python interpreter as the external tool. Our results show that the TIR model decisively outperforms its pure-text counterpart on the pass@𝑘 metric. Crucially, this advantage is not confined to computationally-intensive problems but extends to those requiring significant abstract insight. We further identify the emergent cognitive patterns that illustrate how models learn to think with tools. Finally, we report improved tool usage behavior with early code invocation and much more interactive turns with ASPO. Overall, our work provides the first principled explanation for TIR’s success, shifting the focus from the mere fact that tools work to why and how they enable more powerful reasoning.

### 1. Introduction

Large language models (LLMs) have rapidly progressed from fluent generators to general-purpose problem solvers. Nevertheless, purely text-based reasoning often struggles with tasks that demand precise calculation, long-horizon search, faithful verification, or access to information beyond a model’s parametric memory. As a powerful and empirically successful paradigm, Tool-Integrated Reasoning (TIR) [3, 8] has emerged to address these limitations. Systems equipped with external tools have consistently and significantly outperformed their pure-text counterparts[9, 10, 18]. However, despite the widespread recognition of TIR’s effectiveness, a principled account of the fundamental mechanisms, specifically why and when it helps, is still missing. Existing research has largely focused on demonstrating empirical success, leaving a crucial gap for a formal framework that can elucidate the origins of its benefits and define its capability boundaries.

To build such a framework, we first turn to reinforcement learning (RL) [6, 13], the predominant paradigm for enhancing LLM reasoning. Recent theoretical work has established a critical consensus: in a pure-text environment, RL is constrained by an “invisible leash” [17]. The learning process is largely confined to re-weighting probabilities within the base model’s pre-existing trajectories, meaning it cannot discover fundamentally new reasoning trajectories that lie outside this initial capability [22].

The central thesis of this work is that tool integration fundamentally breaks this barrier. By introducing deterministic, non-linguistic state transitions via an external tool like a Python interpreter, TIR fundamentally expands the model’s exploratory space. We provide the first formal proof that TIR enables a strict expansion of the model’s empirical support, allowing it to generate correct trajectories that have

Corresponding author: zhongwenxu@tencent.com

negligible or even zero probability in a pure-text paradigm. Beyond theoretical reachability, we introduce the concept of token efficiency to argue that tools are a practical necessity. For any finite token budget, there exist algorithmic strategies whose programmatic representations are concise, while their naturallanguage simulations are intractably verbose. Consequently, TIR unlocks a vastly larger feasible support of problem-solving strategies that are simply out of reach for pure-text models under realistic constraints. Extensions to other tools with informal propositions can be found in Appendix A.

We validate these theoretical claims through a series of experiments focusing on solving mathematical competition problems with a Python code interpreter. Our pass@𝑘 analysis provides clear evidence that TIR decisively breaks the capability ceiling of pure-text models. Further investigation, using our proposed “algorithmic friendliness” metric, reveals that TIR’s benefits are not confined to computationally-intensive problems but extend to those requiring significant abstract insight. Case studies of the model’s behavior further illuminate how it leverages this expanded capability, revealing three emergent cognitive patterns: insight-to-computation transformation, exploration & verification via code, and offloading of complex calculations.

Finally, in exploring how to further optimize TIR models, we identify a practical algorithmic challenge: guiding model behavior, such as encouraging earlier tool use, via traditional reward shaping often leads to training instability in GRPO-like algorithms [3, 12]. To address this, we propose Advantage Shaping Policy Optimization (ASPO), a novel algorithm that circumvents the reward function and instead applies a stable, controllable bias directly to the advantage function. Our experiments show that ASPO successfully guides model behavior with early tool invocation and increased tool usages without compromising task performance or training stability.

Our contributions are as follows:

- 1. We provide the first formal theory for why TIR expands an LLM’s capabilities, proving that it enables a strict expansion of both the feasible and empirical support compared to pure-text models.
- 2. We propose Advantage Shaping Policy Optimization (ASPO), a novel and stable algorithm for guiding the behavior of TIR models by directly shaping the advantage function, overcoming the instability of traditional reward-based methods.
- 3. We conduct a comprehensive empirical analysis that not only validates our theoretical claims and algorithm but also provides a mechanistic explanation of TIR’s effectiveness, identifying its universal benefits across problem types and the emergent cognitive patterns it fosters.

### 2. Related Work

A significant body of work focuses on developing RL frameworks for strategic tool use. Feng et al. [3] propose ReTool, an RL-based framework that demonstrates high data efficiency for learning tool use compared to general-purpose reasoning approaches like Yu et al. [21]. Similarly, Li et al. [8] introduce ToRL, a method designed to address the challenges of scaling tool-integrated RL to more complex and demanding scenarios. Bai et al. [1] document methods for effective code-integrated reasoning, where models generate and execute code to arrive at solutions. This paradigm shares the goal of augmenting LLM reasoning with external, verifiable Python execution. Focusing on training from base models, Xue et al. [20] present SimpleTIR, an end-to-end framework for multi-turn TIR that enables stable training from scratch, a process they refer to as the “Zero” setting.

While these methods show empirical success, other research investigates the theoretical limitations and fundamental effects of RL on LLM reasoning. Yue et al. [22] empirically question whether RL truly incentivizes novel reasoning capacity or merely optimizes the exploitation of the base model’s pre-existing abilities. Providing a theoretical framework that helps explain such empirical findings, Wu et al. [17] analyze why RLVR may be constrained by the base model’s initial capabilities. Their “invisible leash” theory suggests that models may struggle to discover reasoning paths far from their original knowledge distribution.

Beyond programmatic tools like Python interpreters, another significant line of research focuses on integrating search engines to provide LLMs with up-to-date, external knowledge through reinforcement learning. Jin et al. [5] introduce Search-R1, an RL framework where LLMs learn to autonomously interleave reasoning steps with real-time search engine queries. The model is optimized using a simple outcome-based reward, and its training is stabilized by masking losses on retrieved tokens, demonstrating strong performance on multi-turn question-answering tasks. Addressing the challenge of navigating extreme uncertainty in complex web-based tasks, Li et al. [7] introduce WebSailor, a complete posttraining methodology designed to close the performance gap with proprietary agents, achieve stateof-the-art results on challenging information-seeking benchmarks like BrowseComp. In this work, we primarily focus on utilizing Python interpreters to enhance the LLM’s ability to solve complex reasoning problems in mathematics; similar principles apply for enhancing knowledge-seeking ability and we have informal discussions in Appendix A.

### 3. Method

In this section, we formalize the argument that integrating an external computational tool, such as a code interpreter, fundamentally enhances a Large Language Model’s (LLM) capabilities. We structure our argument in two parts. First, we provide a formal proof demonstrating that tool integration results in a strict expansion of the model’s generative support, thereby breaking the “invisible leash” that constrains purely text-based models [17]. Second, we introduce the concept of token efficiency to argue that even for problems theoretically solvable by text-based models, tool integration is a practical necessity for expressing complex algorithms within any feasible token budget.

##### 3.1. Formal Proof: Support Expansion via Tool Integration

We begin by establishing that augmenting an LLM with a deterministic external tool strictly expands its support, enabling it to generate trajectories that were previously impossible.

##### 3.1.1. Theoretical Context: The Limits of Standard RL

To ground our proof, we first adopt the theoretical framework proposed by Wu et al. [17], which formalizes the limitations of standard on-policy reinforcement learning [6, 11, 14] on training LLMs.

- Definition 3.1 (Support of a Model (adapted from [17])). Let 𝒴 be the space of all possible generative trajectories. The support of a model with distribution 𝑝(𝑦|𝑥) is the set of all trajectories that can be generated with a non-zero probability for a given prompt 𝑥:

supp(𝑝) := {𝑦 ∈ 𝒴 | 𝑝(𝑦|𝑥) > 0}

- Definition 3.2 (Empirical Support (from [17])). For a threshold 𝜀 > 0, define the empirical support of 𝑝 as

supp𝜀(𝑝) := {𝑦 ∈ 𝒴 | 𝑝(𝑦|𝑥) ≥ 𝜀}.

This definition is central to understanding a model’s intrinsic capabilities. The following theorem from Wu et al. [17] establishes a key constraint for models trained with Reinforcement Learning from Verifiable Rewards (RLVR) [6, 14]:

Theorem 3.3 (Support Preservation under RLVR (from [17])). Let 𝜋𝜃(𝑦|𝑥) be an RLVR-trained policy distribution initialized from a base model with distribution 𝑞(𝑦|𝑥). For any prompt 𝑥, the support of the trained policy is a subset of the support of the base model:

supp(𝜋𝜃) ⊆ supp(𝑞) This implies that if 𝑞(𝑦*|𝑥) = 0 for a correct trajectory 𝑦*, then RLVR can never discover 𝑦*.

Theorem 3.3 formalizes the “invisible leash”: RLVR can only re-weight probabilities within the model’s pre-existing support. We next show a strictly stronger, practical statement under an empirical-support view.

##### 3.1.2. Proof of Support Expansion

We consider two types of LLMs in this work. A pure-text model is a standard language model with distribution 𝑞text that generates tokens exclusively from its vocabulary 𝒱. We compare this to a toolintegrated model, a system (𝑀,𝒪) with distribution 𝑝TIR, which pairs a language model 𝑀 with a deterministic external oracle 𝒪 (e.g., a Python interpreter). The generative process for this model includes not only probabilistic token generation from 𝒱 but also deterministic tool-use transitions. In such a transition, the model 𝑀 emits a tool call 𝑦call, the oracle executes it, and the resulting output 𝑦out = 𝒪(𝑦call) is deterministically returned as the next state.

Now we present the main theorem and its proof:

Theorem 3.4 (Strict Expansion of Empirical Support via Tool Integration). There exists an 𝜀 > 0 and a family of problem instances such that

(︀

)︀ ⊂ supp𝜀

(︀

)︀

supp𝜀

𝑝TIR

.

𝑞text

Proof. The proof proceeds in two parts. First, we establish the subset relationship (⊆), and second, we prove the relationship is strict (̸=) by demonstrating the existence of trajectories accessible only to the tool-integrated model.

##### Part 1: Proving supp(𝑞text) ⊆ supp(𝑝TIR)

Let 𝑦 be an arbitrary trajectory in the support of the pure-text model, such that 𝑞text(𝑦|𝑥) > 0. The trajectory 𝑦 consists exclusively of tokens from the vocabulary 𝒱. The tool-integrated model 𝑝TIR can generate this same trajectory by adopting a policy of never invoking the external oracle 𝒪. Since its generative capabilities subsume those of 𝑞text, it can assign a non-zero probability to the trajectory 𝑦. Thus, for any 𝑦 ∈ supp(𝑞text), it follows that 𝑦 ∈ supp(𝑝TIR), establishing that supp(𝑞text) ⊆ supp(𝑝TIR).

##### Part 2: Proving Strictness

To prove strictness, we use a constructive approach based on a standard cryptographic primitive: a random oracle. Let us consider a problem instance where the solution requires computing 𝑦out = 𝐻(𝑥), where 𝐻 is a random oracle. A random oracle is a theoretical black box that, for any new input query, returns an output chosen uniformly at random from its output space (e.g., {0,1}𝑚), but deterministically returns the same output for repeated queries of the same input. This construction is theoretically convenient and serves as an idealization of practical cryptographic hash functions (e.g., SHA-256). For a model without access to the oracle, its only strategy to find 𝑦out is to guess it. The probability of correctly guessing a specific 𝑚-bit string is 2−𝑚.

Now, consider a trajectory 𝑦* = (𝑦prefix* ,𝑦out,𝑦suffix* ) that involves computing 𝐻(𝑥). We assume the underlying language model for both 𝑝TIR and 𝑞text is identical. The tool-integrated model 𝑝TIR can invoke the oracle to obtain 𝑦out deterministically. In contrast, the pure-text model, 𝑞text, must guess 𝑦out from an output space of size 2𝑚, succeeding with a probability of only 2−𝑚. Thus, the total probabilities of producing 𝑦* are directly related:

𝑞text(𝑦*|𝑥) = 𝑝TIR(𝑦*|𝑥) · 2−𝑚.

For any non-negligible probability 𝑝TIR(𝑦*|𝑥) and a sufficiently large 𝑚, the corresponding 𝑞text(𝑦*|𝑥) becomes arbitrarily small. We can therefore always choose an 𝜀 such that 𝑞text(𝑦*|𝑥) < 𝜀 ≤ 𝑝TIR(𝑦*|𝑥). So we find that 𝑦* ∈/ supp𝜀(𝑞text) while 𝑦* ∈ supp𝜀(𝑝TIR). This establishes strictness.

| |
|---|

We have shown that supp(𝑞text) is a strict subset of supp(𝑝TIR). Unlike pure-text models, which are constrained by Support Preservation (Theorem 3.3), tool integration breaks the “invisible leash” by introducing new, deterministic state transitions, thereby creating a strict expansion of the model’s support.

##### 3.2. Token Efficiency and Feasible Support under a Budget

The proof in the previous section establishes that a tool-integrated model can generate trajectories that are impossible for a pure-text model. This, however, raises a deeper question: can a pure-text model achieve the same outcomes by simulating the computational process through natural language? While the resulting trajectories may differ syntactically (𝑦text ̸= 𝑦TIR), they might represent the same underlying problem-solving strategy. To properly evaluate this, we must move beyond comparing trajectories based on string identity and instead assess them on their semantic content and efficiency. This motivates our analysis of token efficiency.

##### 3.2.1. The Concept of Token Efficiency

A key distinction between programmatic and natural language solutions is their token efficiency: the compactness with which a solution is represented. For any task involving iteration or recursion, a programmatic representation offers a scalable, abstract description with a near-constant token cost, e.g., 𝑂(1). In contrast, a natural language trace that simulates the same process must enumerate each computational step, leading to a token cost that scales with the size of the computation. The tables below illustrate this stark disparity for common algorithmic patterns: simple iteration (Table 1), large linear systems (Table 2), dynamic programming (Table 3), and graph search (Table 4). In each case, the

programmatic solution remains a concise, scalable representation, while the natural language simulation becomes a verbose, concrete enumeration that is untenable for non-trivial problem sizes.

- Table 1: Contrasting Token Efficiency for an Iterative Task (𝑁 → ∞)

Programmatic Approach (Python) Natural Language Reasoning

A symbolic, abstract representation of the computation. The token cost is constant and independent of 𝑁.

A concrete, step-by-step enumeration of the computation. The token cost scales with the magnitude of 𝑁.

- 1 # N can be 10,000,000 or more

- 2 for i in range(N):

- 3 # Perform some check

- 4 check(i)

- 5

"Okay, to solve this, I must check every number. First, for n=1, I perform the check...

- Next, for n=2, I perform the check...
- Next, for n=3, I perform the check...

... (This enumeration continues for millions of steps)

... Finally, for n=10,000,000, I perform the check..."

Token Cost: A few dozen tokens. Scales as 𝑂(1). This is highly efficient and scalable.

Token Cost: Proportional to 𝑁. Scales as Ω(𝑁). This is inefficient and becomes intractable for large 𝑁, quickly exceeding any feasible context window.

- Table 2: Contrasting Token Efficiency for Solving Large Linear Systems

##### Programmatic Approach (Python) Natural Language Reasoning

A single call to a highly optimized numerical library solves 𝐴𝑥 = 𝑏. The token cost is constant, independent of the matrix dimension 𝑛.

A detailed explanation of Gaussian elimination, requiring a description of each row operation. The token cost scales with the matrix size.

"To solve the system, we perform Gaussian elimination. First, to eliminate the first variable from the second row, we subtract 𝐴2,1/𝐴1,1 times the first row from the second row. We must do this for all 𝑛 − 1 rows below the first. Next, we use the new second row to eliminate the second variable from the rows below it... (This narration continues for 𝑂(𝑛2) elements and 𝑂(𝑛3) operations)."

- 1 import numpy as np

- 2 # A is a large n x n matrix ,

- 3 # e.g., n=1000

- 4 x = np.linalg.solve(A, b)

Token Cost: A few tokens. Scales as 𝑂(1). Enables solving massive systems within a tiny token budget.

Token Cost: Proportional to the number of elements in the matrix to sketch. Scales as Ω(𝑛2). A full narration would scale as Ω(𝑛3).

##### 3.2.2. Feasible Support under a Token Budget

The fundamental disparity in token efficiency motivates a more practical, budget-aware analysis of a model’s capabilities, moving beyond theoretical possibilities to what is achievable within operational constraints. To formalize this, we first define the total token cost of a trajectory, cost(𝑦), as the sum of all tokens consumed (i.e., prompt, model generation, and tools I/O), which must not exceed the model’s context budget 𝐵. This allows us to define the set of strategies a model can feasibly execute:

- Definition 3.5 (Computational Equivalence Class). Two trajectories, 𝑦1 and 𝑦2, are computationally equivalent, denoted 𝑦1 ∼ 𝑦2, if they solve the same problem 𝑥 by implementing the same core algorithm. This relation partitions the space of all trajectories 𝒴 into equivalence classes, where each class [𝑦] represents a distinct algorithmic “idea” or “strategy”.
- Definition 3.6 (Feasible Support under Budget 𝐵). An algorithmic strategy, represented by equivalence

class [𝑦], is within the feasible support of a model 𝑀 under token budget 𝐵, denoted [𝑦] ∈ supp𝐵(𝑀), if and only if there exists at least one trajectory 𝑦′ ∈ [𝑦] such that 𝑀(𝑦′|𝑥) > 0 and its token cost(𝑦′) does not exceed the budget:

∃𝑦′ ∈ [𝑦] s.t. 𝑀(𝑦′|𝑥) > 0 and cost(𝑦′) ≤ 𝐵.

This definition captures a model’s practical ability to realize a problem-solving strategy within operational constraints. With this formal framework in place, we can now state our central claim regarding the practical supremacy of tool-integrated models:

Theorem 3.7 (Strict Supremacy of Tool-Integrated Feasible Support). For any non-trivial algorithmic problem class, there exists a problem size 𝑛𝐵 such that for any token budget 𝐵, the feasible support of a pure-text model is a strict subset of the feasible support of a tool-integrated model:

supp𝐵(𝑞text) ⊂ supp𝐵(𝑝TIR)

Proof. The proof requires showing both inclusion (⊆) and strictness (̸=). Inclusion (⊆): Any algorithmic strategy that is feasibly executable by a pure-text model within budget 𝐵 is, by definition, also executable by a tool-integrated model that simply abstains from using its tool.

Strictness (̸=): We must show there exists an algorithmic class [𝑦𝐴] in supp𝐵(𝑝TIR) but not in supp𝐵(𝑞text). This follows directly from the divergent scaling properties of natural language versus programmatic representations, as illustrated in Tables 1-4. For any algorithm whose pure-text simulation cost scales with problem size 𝑛 (e.g., Ω(𝑛), Ω(𝑉 + 𝐸)), we can choose a size 𝑛𝐵 such that the cost exceeds any finite budget 𝐵. The programmatic representation, costing 𝑂(1), remains within budget. Thus, for a sufficiently large problem size, the corresponding algorithmic classes are in the feasible support of 𝑝TIR but not 𝑞text, proving strict inclusion.

| |
|---|

The theorem crystallizes the practical implications of token efficiency. It establishes that for any finite computational budget, there is a vast class of algorithmic strategies that pure-text models are fundamentally incapable of executing. Not because the solution is unknowable, but because its expression in natural language is too verbose. Tool integration is therefore not merely a convenience; it is a necessity for expanding the set of algorithmic approaches that LLMs can feasibly deploy. This provides a strong

argument for a paradigm where LLMs act as reasoning engines that delegate complex computational tasks to specialized, efficient tools.

##### 3.3. Algorithmic Improvement: Advantage Shaping for Early Code Invocation

The TIR models often default to a conservative strategy: completing the majority of their abstract reasoning via text before invoking the code interpreter for the final-step calculation or verification. This overlooks a potentially more powerful paradigm where the interpreter is used as an exploratory tool throughout the reasoning process. We hypothesize that encouraging the model to invoke code earlier could foster a more dynamic, flexible, and hypothesis-driven reasoning style, potentially unlocking novel problem-solving strategies.

Our initial, most direct approach was to introduce an early-code reward directly into the reward function. For each response 𝑖 that is both correct and code-containing in a group of samples, we added a reward term 𝑟𝑖′ that penalizes later code invocation:

𝑅𝑖 = 1 + 𝑟𝑖′ where 𝑟𝑖′ = 𝛿 · clip(︂

,−𝑐,𝑐)︂. (1)

𝑝𝑖 − mean(p) std(p)

However, this seemingly innocuous modification proved to be highly destabilizing during training (see experimental details in Section 4.5 and Figure 5 (a)). In algorithms like GRPO that rely on group normalized advantage, this design has a critical flaw. In the common scenario where all samples in a group are correct, the primary reward signal (the constant ‘1’) is entirely eliminated by the normalization. The advantage calculation then becomes:

𝑅𝑖 − mean(R) std(R)

(1 + 𝑟𝑖′) − (1 + mean(r′)) std(r′)

𝑟𝑖′ − mean(r′) std(r′)

𝐴𝑖 =

=

=

This leads to a catastrophic outcome: (1) the primary signal about answer correctness disappears, (2) the auxiliary signal 𝑟𝑖′ is amplified to the same magnitude as the original primary signal, and (3) due to the nature of standardization, approximately half of these correct responses receive a negative advantage and are thus heavily penalized, solely because their code invocation is later than the group’s average.

To circumvent the distorting effects of reward normalization, we developed a more robust method that we term Advantage Shaping Policy Optimization (ASPO). Instead of manipulating the reward, we directly modify the final advantage value after the standard correctness-based advantage 𝐴correct has been calculated. For any response 𝑖 that is both correct and contains code, we compute the new advantage 𝐴𝑖 as follows:

, −𝑘 · 𝐴correct,𝑖, 𝑘 · 𝐴correct,𝑖)︂, (2)

𝐴𝑖 = 𝐴correct,𝑖 + clip(︂𝛿 ·

𝑝𝑖 − mean(p) mean(L)

where p and L are the sets of first code invocation positions and total response lengths for all correct, code-containing responses within the group. Furthermore, 𝛿 is a negative coefficient to encourage early code invocation, and 𝑘 is a clipping hyperparameter that bounds the magnitude of auxiliary advantage within a proportion of the basic advantage of correctness.

This formulation has several key merits, primarily by circumventing the uncontrollable effects of advantage normalization inherent to reward-based modifications. First, it addresses a critical flaw in the rewardbased approach: the inability to guarantee a positive advantage for all correct answers. After adding

the auxiliary reward, a correct response’s total reward could fall below the group average, leading to a negative GRPO normalized advantage, which effectively punishes a correct solution. Second, the GRPO normalization process itself introduces uncontrollable volatility: the std(R) in the denominator unpredictably scales the auxiliary signal, making its influence inconsistent across different groups.

Our ASPO algorithm resolves both issues. By applying a clipped bias directly to 𝐴correct, we ensure the final advantage remains positive and that the early-code incentive is always a subordinate nudge, never overwhelming the primary objective of correctness. Furthermore, this approach bypasses the volatile scaling effect of std(R) entirely. Finally, the choice to normalize the code invocation position by the mean response length mean(L) rather than the standard deviation of positions std(p) is deliberate. The latter is unstable: when invocation positions in a group are tightly clustered, a small std(p) would excessively amplify the signal, whereas a more stable denominator like mean(L) is consistent and meaningful. This method allows us to stably and effectively encourage early code invocation, the empirical results of which are detailed in Section 4.5.

In essence, ASPO provides a general and robust framework for guiding a model’s behavior towards desired styles or properties without compromising the primary learning objective (e.g., accuracy). By directly manipulating the advantage values, ASPO avoids the instabilities that can arise from altering the reward function, particularly in GRPO-like algorithms that rely on reward normalization. This method ensures that the incentive for the desired behavior (in this case, earlier code invocation) acts as a stable adjustment. The core principles of ASPO could be readily adapted to encourage other desirable behaviors in a variety of scenarios, offering a reliable approach to shape model conduct while preserving training stability and overall task performance.

### 4. Experiments

##### 4.1. Experimental Setup

(a) Training accuracy

0.90

Accuracy(%)

0.85

0.80

TIR

Pure-text

0.75

0 20 40 60 80

Training steps

(b) Test accuracy

AIME25Accuracy(%)

TIR

0.65

Pure-text

0.60

0.55

0 20 40 60 80

Training steps

- Figure 1: The (a) training and (b) testing accuracy of the TIR and pure-text RL on Qwen3-8B model. The AIME25 accuracy (b) is the average of 16 responses.

Model and Datasets. All experiments are based on the Qwen3-8B model [16]. For our training data, we randomly sample 10,000 English problems from the DAPO dataset [21] due to limited computational resources. Since our aim is to fundamentally understand the mechanisms of TIR rather than to improve absolute accuracy of benchmarks, this dataset is sufficient for our purpose, in contrast to the extensive training datasets used in other literature [3, 21]. Our primary evaluation benchmarks are AIME24,

AIME25, and a challenging subset of the Omni-MATH dataset [4]. For the latter, due to the large size of the dataset, we curated the 512 most difficult problems that are amenable to reliable, rule-based evaluation, which we denote as Omni-MATH-512.

Training Protocol. We train two main models for comparison: our proposed TIR model, which can execute code to assist in its reasoning process, and a pure-text RL model as a baseline (as shown in Figure 1). Both models are trained for 3 epochs using the DAPO algorithm [21], a variant of GRPO [14]. During training, we use a rollout batch size of 96 problems, with 8 responses sampled per problem, a maximum response length of 16,384 tokens, and a sampling temperature of 1.0 to encourage exploration.

Evaluation Protocol. For evaluations, we set the sampling temperature to 0.6 and maximum response length to 16,384 tokens unless otherwise specified.

##### 4.2. Pass@𝐾 Experiments: TIR Breaks the Capability Ceiling

To empirically test our theoretical claims, this section investigates whether TIR can overcome the capability ceiling observed in pure-text models [17, 22]. Similar to Yue et al. [22] and others, we use the pass@𝑘 metric, with low-variance estimation from Chen et al. [2], as it provides a robust measure of a model’s underlying problem-solving potential.

- Figure 2 presents the macroscopic evidence from our experiments. It plots the pass@𝑘 curves for both the TIR model (RL trained) and the pure-text baseline (Qwen3 8B) across our three evaluation benchmarks, with the max 𝑘 of 256. The results are unequivocal: on AIME24, AIME25, and Omni-MATH-512, the TIR model’s performance curve is consistently and significantly higher than that of the pure-text model. Crucially, we observe no intersection between the curves, even as 𝑘 increases to 256. This stands in stark contrast to previous findings where RL-trained text models, while improving pass@1, often do so at the cost of the broader capability envelope, eventually being surpassed by the base model at high values of 𝑘 [22]. Our results show that TIR does not suffer from this trade-off; it elevates the entire pass@𝑘 curve.

To further understand this performance gain at a per-problem level, we visualize the “flow of solvability” on the Omni-MATH-512 dataset in Figure 3. This Sankey diagram illustrates how the solvability status of individual problems changes when moving from the pure-text model to the TIR model (samples

1.0

0.9

pass@k

0.8

0.7

0.6

(a) AIME24

TIR

Pure-text

1 2 4 8 16 32 64 128 256

Number of samples k

1.0 (b) AIME25

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.9

0.8

0.7

0.6

0.5

1 2 4 8 16 32 64 128 256

Number of samples k

(c) Omni-MATH (512)

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.7

0.6

0.5

0.4

1 2 4 8 16 32 64 128 256

Number of samples k

- Figure 2: Pass@𝑘 curves for the TIR (RL trained) and pure-text models (Qwen3 8B) across three benchmarks: (a) AIME24, (b) AIME25, and (c) Omni-MATH-512. The detailed numerical data corresponding to this figure are provided in the Appendix D.

###### Flow of Problem Solvability

###### Pure-text TIR

Unsolved 18.0%

Unsolved 31.6%

Jointly Unsolved 16.2%

Expansion 15.4% Shrinkage 1.8%

Solved 82.0%

Solved 68.4%

Preservation 66.6%

- Figure 3: The flow of problem solvability on Omni-MATH-512 when transitioning from the pure-text model to the TIR model, evaluated at 𝑘 = 256.

𝑘 = 256 responses per problem). We categorize the problems into four distinct groups as [17]: Capability Expansion: Problems the pure-text model fails to solve but the TIR model succeeds on; Capability Preservation: Problems solved by both models; Capability Shrinkage: Problems solved by the pure-text model but not by the TIR model; Jointly Unsolved: Problems that neither model can solve. The diagram reveals a massive net gain in problem-solving capability. The Capability Expansion set contains 15.4% problems, whereas the Capability Shrinkage set contains only 1.8%. This provides direct empirical validation for our theoretical argument in Section 3, demonstrating that TIR facilitates a practically significant expansion of the model’s effective support.

In summary, both macroscopic pass@𝑘 analysis and microscopic problem-level tracking confirm that tool-integrated reasoning decisively breaks the capability ceiling of its pure-text counterpart, enabling the model to solve a wide range of problems that were previously out of its reach.

##### 4.3. Benefits of TIR Extend Beyond Computationally-Intensive Problems

A crucial question arises from our initial findings: is the observed capability expansion of TIR merely an artifact of solving problems that are inherently algorithmic? The most direct yet naive interpretation of TIR’s success is that it simply offloads complex arithmetic, acting as a superior calculator. However, a more nuanced counterargument posits that TIR’s effectiveness, while beyond simple calculation, is still confined to problems whose structure can be directly mapped to a known algorithm such as exhaustive search in combinatorics. This perspective suggests that TIR improves the model’s capability on problems that are computationally-intensive or inherently algorithmic, but offers little advantage when the problem is highly abstract.

To rigorously test our hypothesis, we first introduce the concept of “algorithmic friendliness”, which is defined as a measure of how reliant a problem’s solution is on standard computation versus deep mathematical insight. To operationalize this concept, we developed a detailed five-point rubric for

0.8

pass@k

0.6

0.4

0.2

0.8

pass@k

0.6

0.4

0.2

(a) Algo friendliness G1

TIR

Pure-text

1 2 4 8 16 32 64 128 256

(d) Algo friendliness G4

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

1 2 4 8 16 32 64 128 256

Number of samples k

(b) Algo friendliness G2

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.6

0.4

0.2

1 2 4 8 16 32 64 128 256

(e) Algo friendliness G5

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.6

0.4

0.2

1 2 4 8 16 32 64 128 256

Number of samples k

0.8

0.6

0.4

0.2

Percentageofproblems(%)

30

20

10

(c) Algo friendliness G3

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

1 2 4 8 16 32 64 128 256

(f) Dataset Omni-MATH-512

| |24<br><br>141<br><br>157<br><br>120<br><br>70|
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0

G1

G2

G3

G4

G5

Algo friendliness group

- Figure 4: (a)-(e) Pass@k curves for the TIR and pure-text models, grouped by problem algo friendliness. (f) The distribution of algo friendliness scores across the Omni-MATH-512 dataset. The problems are categorized into five groups based on their algo friendliness scores: 1.0–1.5 (G1), 2.0–2.5 (G2), 3.0–3.5 (G3), 4.0–4.5 (G4), and 5.0 (G5).

classifying problems, as presented in Appendix C. This scale ranges from a score of 1 for problems that are fundamentally abstract and non-computational, to 5 for those solvable by a direct application of a textbook algorithm. We then applied this rubric to classify each problem in the Omni-MATH-512 dataset. This classification was performed by providing both the problem statement and its solution idea to the Gemini 2.5 Pro API [15], which then assigned a score based on the rubric. The resulting distribution of problem types, shown in Figure 4(f), reveals a crucial characteristic of the dataset. Contrary to being biased towards highly algorithmic problems, the distribution’s peak is concentrated in the medium friendliness categories (scores 2, 3 and 4). This confirms that Omni-MATH-512 serves as a fair and challenging testbed for our analysis, not one skewed towards problems with simple computational solutions.

Figure 4(a)-(e) presents our core findings. It displays the pass@𝑘 curves for the TIR and pure-text models, grouped by the algo friendliness of the problems. As expected, the performance gap between the two models is most pronounced for problems with high friendliness (scores 4.0-5.0), where TIR’s ability to execute algorithms directly provides a massive advantage (Figure 4(a),(b)). The most critical finding, however, comes from the lowest friendliness group (scores 1.0-2.5). Even for these problems, which depend heavily on abstract reasoning and are ill-suited to direct computation, the TIR model maintains a significant and consistent performance advantage over the pure-text baseline, outperforming it by approximately 9% in pass@256 accuracy (Figure 4(d),(e)).

This result demonstrates that the benefits of TIR are not confined to easily programmable problems. The tool serves a more profound purpose than acting as a simple calculator or a direct algorithm-implementer. It suggests that the model is leveraging the code interpreter in more complex and sophisticated ways,

which we will investigate in the next subsection.

##### 4.4. Case Analysis: Emergent Cognitive Patterns of Tool Integration

The quantitative results of the previous sections demonstrate that TIR is universally effective, but they do not fully explain how. If the model’s advantage is not limited to algorithmically amenable problems, how exactly is it leveraging the code interpreter to solve problems requiring abstract insight? Through qualitative analysis of model outputs, we have identified three distinct and recurring patterns of code utilization that answer this question.

- Pattern 1: Insight-to-computation transformation. In this primary pattern, the model’s first step is not to code, but to reason. It engages in non-trivial, text-based analysis to deconstruct a complex problem, applying mathematical insights to transform it into a state that is amenable to a programmatic solution. The code interpreter is then invoked to execute a genuine algorithm (such as search, enumeration, or DP) that efficiently solves this newly formulated sub-problem under a limited computational resource. Unlike straightforward sequential calculations one might perform with a calculator, these algorithms often possess complex control flows (e.g., loops, recursion) that are challenging for a language model to emulate or follow step-by-step.

For instance, as shown in Table 7, the model first uses mathematical reasoning to derive a transcendental equation from the abstract geometric problem. It then employs code to iterate the entire parameter space of (𝑚,𝑛) pairs, using the Intermediate Value Theorem as a numerical method to efficiently detect whether a solution exists for each pair.

- Pattern 2: Exploration and verification via code. For problems where the solution path is not immediately obvious, the model utilizes the code interpreter as an interactive sandbox for exploration and hypothesis testing. Instead of committing to a single line of reasoning, it formulates conjectures and writes short code snippets to test them, observe their outcomes, and iteratively refine its strategy based on the feedback. This pattern is particularly prevalent in problems with low algorithmic amenability, where it allows the model to build confidence and discover insights through empirical experimentation.

Table 8 provides a clear instance of this exploratory behavior. The model first derives a candidate value of 𝜆 = √3 from a simple case, then uses the code interpreter to numerically explore more different scenarios. The feedbacks validate its initial hypothesis and pivot its strategy from further exploration toward constructing a rigorous algebraic proof.

These first two patterns represent a fundamental departure from pure-text reasoning. As we established in Section 3, they constitute entirely new Computational Equivalence Classes, new strategies for solving problems. While a pure-text model might theoretically be able to simulate these processes, the token cost of doing so would be astronomical. The step-by-step, trial-and-error nature of the exploratory pattern, in particular, would lead to a blow-up in token length. Therefore, these strategies lie far outside the Feasible Support under Budget 𝐵 for any practical context window, making them uniquely accessible to the TIR paradigm.

- Pattern 3: Offloading complex calculation. This is the most direct pattern of tool use, where the model has a clear, linear path to the solution but delegates complex or tedious calculations to the interpreter. This usage aligns with the naive view of TIR as a “calculator”, but its importance should not be understated. By offloading these steps, the model minimizes the risk of unforced computational errors that frequently derail long chains of pure-text thought, thereby preserving the integrity of the overall reasoning process.

A representative example is shown in Table 9. Here, the model first performs the text-based reasoning to establish a solution path, then uses the interpreter as a precision tool to execute the series of intricate vector and algebraic computations that would be highly prone to manual error.

In conclusion, these emergent patterns reveal a sophisticated interplay between the LLM’s reasoning capabilities and the code interpreter’s computational power. The model is not merely using a tool; it is thinking with tools. This signifies a fundamental shift in strategy: rather than simply delegating calculations from an otherwise unchanged, text-based line of thought, the model learns to generate novel problem-solving approaches that are intrinsically synergistic with the interpreter. It formulates plans that leverage programmatic strengths like iteration and DP from the outset, developing new “Computational Equivalence Classes” that were previously inaccessible. Such dynamic and flexible code invocation enables the TIR model to break the capability ceiling of its pure-text counterpart.

##### 4.5. Empirical Analysis of ASPO for Early Code Invocation

In this section, we empirically validate our ASPO algorithm, designed to encourage earlier code invocation. We aim to answer two primary questions: (1) Does this method maintain training stability and final task performance, unlike the naive reward-based approach? (2) Does it effectively and controllably alter the model’s tool-use behavior as intended? We test our baseline model against the unstable reward-based approach and two variants of our ASPO algorithm: a conservative setting (𝛿 = −2.0,𝑘 = 0.7) and an aggressive setting (𝛿 = −2.5,𝑘 = 0.9).

(a) Training accuracy

(b) Test accuracy (AIME25 avg@16)

0.90

0.675

0.650

0.85

0.625

0.80

0.600

0.75

0.575

0.70

Reward-based Approach

0.550

0 20 40 60 80

0 20 40 60 80

Training steps

Training steps

Baseline ASPO ( = 1.5, k = 0.7) ASPO ( = 2, k = 0.9)

- Figure 5: The (a) training and (b) testing accuracy of the baseline and ASPO algorithm.

Stability and performance remain uncompromised. Figure 5 provides a clear validation of our method’s stability. As mentioned in our analysis in Section 3, the naive reward-based approach quickly becomes unstable, causing the training reward to collapse (Figure 5(a)). In stark contrast, the training curves for our ASPO algorithm with both conservative and aggressive settings remain stable and almost perfectly aligned with the baseline. Furthermore, this stability does not come at the cost of final performance. Figure 5(b) shows that the final “avg@16” accuracy on AIME25 for both variants is statistically indistinguishable from the baseline. This is a crucial result: our method successfully avoids the pitfalls of reward modification, ensuring that the primary goal of solving the problem correctly is not sacrificed.

A significant shift in cognitive behavior. Having established the method’s safety, we now demonstrate its effectiveness in reshaping the model’s reasoning strategy. Figure 6 presents a comprehensive analysis

of the model’s code-use behavior on AIME25, averaged over 16 responses per question. The results show a dramatic and targeted shift. The most significant change is in the code invocation timing (Figure 6(b)), where the average position of the first code call is brought forward from 4,000 tokens in the baseline down to 1,000 tokens. Concurrently, the model becomes a much more active tool user: the average number of code rounds per problem more than doubles, from 1.3 to 3.3 (Figure 6(e)), and the code ratio approaches nearly 100%, indicating that using the interpreter becomes a default part of the model’s process (Figure 6(c)). This behavioral shift is starkly evident when examining the distribution of responses for a single challenging problem. For instance, on Q30 of the AIME25, the baseline model exhibited reluctant and inconsistent tool use: out of 16 independent responses, four failed to make a single code call, and the median number of invocations was just 2. In stark contrast, our ASPO-trained model integrated the tool as an indispensable part of its problem-solving process. It invoked the code in all 16 responses for the same problem, and the median number of tool calls increased from 2 to 13. More significantly, a quarter of the responses demonstrated highly iterative behavior, making more than 20 tool calls, which is entirely absent in the GRPO-trained baseline. This shows a clear transformation from a conservative, late-stage “calculator” usage pattern to an early, iterative, and exploratory “interactive partner” paradigm.

Controllability and absence of reward hacking. Importantly, this behavioral shift is achieved without inducing reward hacking. We manually inspected a large number of samples and found no instances of the model inserting trivial or meaningless code early in its response merely to satisfy the incentive. The stability of the final task accuracy (Figure 5(b)) and the code pass ratio (Figure 6(d)) further substantiates this. Finally, the difference between the conservative and aggressive settings demonstrates that the degree of behavioral change is tunable via the hyperparameters 𝛿 and 𝑘.

### 5. Conclusions

In this work, we presented a comprehensive investigation into the foundational mechanisms of ToolIntegrated Reasoning (TIR). We moved beyond empirical demonstrations to establish a formal theoretical framework explaining its effectiveness. Our core theoretical contribution is the proof that TIR enables a strict expansion of both the empirical and feasible support of an LLM, breaking the “invisible leash” that constrains pure-text models and making complex algorithmic strategies practically achievable within finite token budgets. On the algorithmic front, we identified the instability of reward shaping for guiding model behavior in TIR systems and proposed Advantage Shaping Policy Optimization (ASPO), a stable and effective alternative that directly modifies the advantage function.

Our experiments provided strong empirical validation for these claims. We demonstrated that TIR model equipped with a Python interpreter decisively surpasses the performance of pure-text models across challenging mathematical reasoning benchmarks. Our analysis, using a novel “algorithmic friendliness” metric, revealed that TIR’s benefits are universal, extending even to problems that are highly abstract and less amenable to direct computation. Qualitative analysis further uncovered the sophisticated, emergent cognitive patterns that arise from the synergy between LLM reasoning and tool execution.

Ultimately, our findings advocate for a paradigm shift: viewing LLMs not as monolithic problem-solvers, but as core reasoning engines that intelligently delegate computational tasks to specialized, efficient tools. The principles and methods developed here, particularly ASPO, open avenues for more nuanced and stable control over the behavior of powerful tool-integrated agents.

(a) Response length

12000

11000

10000

9000

8000

7000

0 20 40 60 80

(b) Code invocation timing

7000

6000

5000

4000

3000

2000

1000

0 20 40 60 80

(c) Code ratio

1.0

0.8

0.6

0.4

0 20 40 60 80

(d) Code pass ratio

0.90

0.85

0.80

0.75

Baseline

0.70

- = 1.5, k = 0.7

- = 2, k = 0.9

0.65

0 20 40 60 80

Training steps

(e) Code rounds

3.0

2.5

2.0

1.5

1.0

0.5

0 20 40 60 80

Training steps

(f) Code line

30

25

20

15

0 20 40 60 80

Training steps

- Figure 6: Evaluation results of the baseline and ASPO algorithm on AIME25. (a) Response length, (b) code invocation timing, (c) code ratio, (d) code pass ratio, (e) code rounds and (f) code lines.

### References

- [1] Fei Bai, Yingqian Min, Beichen Zhang, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Towards effective code-integrated reasoning. arXiv preprint arXiv:2505.24480, 2025.
- [2] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [3] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. ReTool: Reinforcement learning for strategic tool use in LLMs. arXiv preprint arXiv:2504.11536, 2025.
- [4] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985, 2024.
- [5] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [6] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman,

- Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tülu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [7] Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, et al. WebSailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025.
- [8] Xuefeng Li, Haoyang Zou, and Pengfei Liu. ToRL: Scaling tool-integrated RL. arXiv preprint arXiv:2503.23383, 2025.
- [9] OpenAI. Introducing GPT-5. Blog post, Aug 2025. URL https://openai.com/index/gpt-5/. Accessed on August 22, 2025. [10].
- [10] OpenAI. Introducing o3 and o4-mini. Blog post, April 2025. URL https://openai.com/index/ introducing-o3-and-o4-mini/. Accessed on August 22, 2025. [10].
- [11] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [12] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [13] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [14] DeepSeek Team. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [15] Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [16] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [17] Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why RLVR may not escape its origin. arXiv preprint arXiv:2507.14843, 2025.
- [18] xAI. Grok 4. Blog post, Jul 2025. URL https://x.ai/blog/grok-4. Accessed on August 22,

2025. [11, 13].

- [19] Zhongwen Xu, Xianliang Wang, Siyi Li, Tao Yu, Liang Wang, Qiang Fu, and Wei Yang. Agents play thousands of 3D video games. arXiv preprint arXiv:2503.13356, 2025.
- [20] Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Zejun Ma, and Bo An. SimpleTIR: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. https://simpletir.notion. site/report, 2025. Notion Blog.
- [21] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [22] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

### A. Extensions to Other Tools and Interactions with Environments

Our arguments in Sections 3.1.2 and 3.2 extend beyond Python to a broad family of external tools and interactive settings. At a high level, any interface that (i) affords state transitions not expressible by next-token sampling alone and/or (ii) delivers high information per token of I/O will both expand support (Section 3.1.2) and strictly enlarge feasible support under a token budget (Section 3.2).

Search and Retrieval Agents. Consider web search, retrieval APIs, or domain databases (e.g., scholarly indices, code search). Let an external retriever implement a (possibly stochastic) mapping ℛ : (𝑞,𝑠) ↦→ 𝑟, where 𝑞 is a query issued by the LLM and 𝑠 is the (latent) world/index state at the time of the call. Even when ℛ is not perfectly deterministic, the trajectory that includes the returned snippet 𝑟 is unreachable for a pure-text model unless it guesses the salient facts in 𝑟 token-by-token. This mirrors the random-oracle argument in Theorem 3.7: as the entropy of 𝑟 conditioned on (𝑞,𝑥) grows, the probability that a pure-text model reproduces 𝑟 by chance decays exponentially, while a tool-augmented model obtains 𝑟 via a single call. Hence support expands, and under any fixed budget 𝐵 the feasible set also strictly expands once the text-only paraphrase of 𝑟 would exceed 𝐵.

Checkers, Verifiers, and Program Runners. Beyond “heavy” computation, many tools act as verifiers: unit tests, symbolic algebra checkers, SAT/SMT solvers, theorem provers, type checkers, or even a Python REPL used only to validate a candidate answer. Such tools add deterministic pruning transitions to the trajectory graph: incorrect branches are cut immediately with 𝑂(1) tokens. This reduces the exploration burden under RLVR-style training and enlarges the set of practically reachable strategies under a budget.

Stateful External Memory. Tools can expose memory larger and more persistent than the model’s context: key–value caches, external scratchpads, vector stores, or file systems. Each call updates an external state 𝑚𝑡+1 = 𝑈(𝑚𝑡,𝑎𝑡) and reads views 𝑣𝑡 = 𝑉 (𝑚𝑡) at 𝑂(1) token cost. Strategies that require memory |𝑚| ≫ 𝐵 are impossible to realize faithfully in pure text (which must inline 𝑚), but become feasible when memory lives outside the context window.

- Proposition A.1 (Informal; External State as Unbounded Scratchpad). Suppose an algorithm requires Ω(𝑛) writable memory cells for problem size 𝑛. If a tool exposes these cells with per-step I/O 𝑂(1), then for

sufficiently large 𝑛, the algorithm’s equivalence class lies in supp𝐵(𝑝TIR) but not in supp𝐵(𝑞text) for any fixed 𝐵.

Embodied and Interactive Environments. When the LLM acts in an MDP or game environment [19], the environment transition 𝑠𝑡+1 = 𝐸(𝑠𝑡,𝑎𝑡) is itself an external oracle. Our earlier support-expansion argument applies verbatim: trajectories that include specific environment observations or states are unreachable by text-only generation unless they are guessed token-by-token. Token-efficiency arguments also lift: environment interactions can realize long-horizon plans with summarized textual traces, whereas a pure-text simulation would require enumerating each counterfactual step.

Noisy or Non-Deterministic Tools. Stochastic returns (e.g., fluctuating search rankings) do not invalidate support expansion. What matters is the existence of some positive-probability outputs with substantial conditional entropy that are infeasible to reproduce via text within budget. In other words, determinism is a convenience, not a necessity, for our conclusions.

Composing Multiple Tools. Real agents chain retrieval, computation, verification, and environment actions. Composition behaves monotonically:

- Proposition A.2 (Informal; Monotone Closure under Composition). Let 𝒯1,...,𝒯𝑘 be tools with per-call costs that sum to at most 𝐵. If each 𝒯𝑖 individually yields a strict feasible-support gain for some subproblem family at size 𝑛𝑖, then there exist composite tasks for which the sequential (or branched) use of {𝒯𝑖} yields a strict feasible-support gain over any pure-text policy at the same total budget.

Takeaway. “Python” is merely one instantiation of a broader principle, our extensions unify code execution, search, verification, memory, and embodied interaction under the same analytical lens.

### B. More Examples on Token Efficiency

- Table 3: Contrasting Token Efficiency for a Dynamic Programming Task (Fibonacci Sequence)

Programmatic Approach (Python) Natural Language Reasoning

A compact representation of the recurrence relation, with a token cost independent of the input integer 𝑁.

A verbose, step-by-step calculation of every subproblem’s solution, with a token cost that grows with 𝑁.

- 1 memo = {0: 0, 1: 1}

- 2 def fib(n):

- 3 if n in memo: return memo[n]

- 4 memo[n] = fib(n-1) + fib(n-2)

- 5 return memo[n]

"To get fib(5), I need fib(4) and fib(3). Fib(2) is fib(1)+fib(0) = 1+0 = 1. Fib(3) is fib(2)+fib(1) = 1+1 = 2. Fib(4) is fib(3)+fib(2) = 3+1 = 4. So, fib(5) is fib(4)+fib(3) = 4+2 = 6... Wait, let me recheck. fib(4) is 3+2=5. No, fib(4) is 2+1=3. Okay, so fib(5) is 3+2=5."

Token Cost: 𝑂(1) Token Cost: Ω(𝑁)

- Table 4: Contrasting Token Efficiency for Search Algorithms

##### Programmatic Approach (Python) Natural Language Reasoning

An abstract procedure for state-space traversal, using data structures like a queue and a set.

A full, step-by-step narration of the entire exploration process, including every node visited and every state change of the queue.

"I start at node ’A’. Queue is [’A’], visited is ’A’. I pop ’A’. Its neighbors are ’B’, ’C’. Queue is now [’B’, ’C’], visited is ’A’,’B’,’C’. I pop ’B’. Its neighbor is ’D’. Queue is now [’C’, ’D’], visited is ’A’,’B’,’C’,’D’. I pop ’C’..." (and so on)

- 1 from collections import deque

- 2

- 3 def bfs(graph , start_node):

- 4 queue = deque([start_node])

- 5 visited = {start_node}

- 6 while queue:

- 7 node = queue.popleft()

- 8 # Process node

- 9 for neighbor in graph[node]:

- 10 if neighbor not in visited:

- 11 visited.add(neighbor)

- 12 queue.append(neighbor)

- 13

Token Cost: Constant cost for the algorithm’s definition. Scales as 𝑂(1).

Token Cost: Proportional to the number of vertices and edges, 𝑉 + 𝐸. Scales as Ω(𝑉 + 𝐸).

### C. Rubric for Algorithmic Friendliness

- Table 5 shows the rubric we use for Gemini Pro APIs [15] to classify the math problems.

- Table 5: Rubric for assessing the “algorithmic friendliness” of problems.

##### Score Level Description Required Insight

5 Very High (Direct Application)

The problem is a textbook example for a standard algorithm (e.g., backtracking). The problem statement itself almost serves as the specification. Almost no mathematical insight is needed.

None beyond basic arithmetic.

4 High (Minor Insight) An algorithm provides a clear advantage, but requires a standard, well-known mathematical identity or simple transformation to be applied. The mathematical hurdle is low.

Recalling and applying a common formula or theorem.

3 Medium (Significant Insight)

A computational solution is effective, but only after applying a significant mathematical insight or performing complex problem modeling. The difficulty is substantial.

A creative, problemspecific trick or a complex modeling effort.

2 Low (Impractical Algorithm)

An algorithm is theoretically possible but highly impractical (enormous search space, precision issues). The algorithmic optimizations are equivalent in difficulty to the mathematical solution.

Insights needed are essentially the mathematical solution itself.

1 Very Low (Noncomputational)

N/A.

The problem is fundamentally abstract and cannot be solved by computation (e.g., requires a formal proof, deals with uncountable sets).

### D. Pass@𝑘 Data

- Table 6 shows the detailed pass@𝑘 results for the TIR and pure-text models across the three benchmarks, evaluated with the max sample size of 256.

- Table 6: Pass@𝑘 results for the TIR model and the pure text model

AIME24 AIME25 Omni-MATH-512 TIR Pure Text TIR Pure Text TIR Pure Text

k

- 1 0.7829 0.6331 0.6841 0.5184 0.5128 0.3585
- 2 0.8408 0.7184 0.7818 0.6065 0.5885 0.4208 4 0.8632 0.7703 0.8395 0.6730 0.6437 0.4707 8 0.8825 0.8050 0.8792 0.7262 0.6869 0.5153 16 0.9024 0.8312 0.9117 0.7613 0.7232 0.5570 32 0.9173 0.8496 0.9339 0.7810 0.7545 0.5942 64 0.9312 0.8645 0.9503 0.7979 0.7802 0.6271 128 0.9480 0.8813 0.9625 0.8250 0.8018 0.6575 256 0.9667 0.9000 0.9667 0.8667 0.8203 0.6836

### E. Capability Expansion and Shrinkage

#### Flow of Problem Solvability

###### Pure-text TIR

Unsolved

###### Share: 18.0%

Unsolved

###### Share: 31.6%

Solved Prob. (0, 0.25]

###### Share: 20.9%

Solved Prob. (0, 0.25]

Solved Prob. (0.25, 0.5]

###### Share: 25.8%

###### Share: 8.8%

Solved Prob. (0.5, 0.75]

Solved Prob. (0.25, 0.5]

###### Share: 8.6%

###### Share: 5.9%

Solved Prob. (0.5, 0.75]

###### Share: 8.2%

Solved Prob. (0.75, 1]

###### Share: 43.8%

Solved Prob. (0.75, 1]

Share: 28.5%

- Figure 7: The detailed flow of problem solvability on Omni-MATH-512 when transitioning from the pure-text model to the TIR model. The solved probability of each problem is evaluated at 𝑘 = 256.

### F. Emergent Cognitive Patterns of Tool Integration

###### [Question]

Let 𝑎 ̸= 𝑏 be positive real numbers and 𝑚, 𝑛 be positive integers. An 𝑚 + 𝑛-gon 𝑃 has the property that 𝑚 sides have length 𝑎 and 𝑛 sides have length 𝑏. Further suppose that 𝑃 can be inscribed in a circle of radius 𝑎 + 𝑏. Compute the number of ordered pairs (𝑚, 𝑛), with 𝑚, 𝑛 ≤ 100, for which such a polygon 𝑃 exists for some distinct values of 𝑎 and 𝑏.

[Response] Okay, let’s try to tackle this problem step by step.

... We need to find the number of ordered pairs (𝑚, 𝑛) where 𝑚, 𝑛 ≤ 100 such that this equation has solutions for 𝑎 and 𝑏 with 𝑎 ̸= 𝑏. ...So, substituting into the equation:

𝑚 · arcsin(𝑥/2) + 𝑛 · arcsin((1 − 𝑥)/2) = 𝜋

... But how can I find all such (𝑚, 𝑛) pairs? Since 𝑚 and 𝑛 are up to 100, perhaps we can write a program to check for each possible pair (𝑚, 𝑛) whether there exists an 𝑥 in (0, 1) such that the equation is satisfied.

But how to implement this check in Python? The problem is that the equation is transcendental and might not have an analytical solution, so numerical methods would be needed. For each (𝑚, 𝑛), we can define a function 𝑓(𝑥) = 𝑚·arcsin(𝑥/2)+ 𝑛 · arcsin((1 − 𝑥)/2) − 𝜋, and check if there’s a root of 𝑓(𝑥) = 0 in (0, 1), excluding 𝑥 = 0.5. But how to check for roots? For example, we can use the Intermediate Value Theorem. If 𝑓(𝑥) changes sign over an interval, there’s a root in that interval.

... Now, let’s proceed to write the Python code.

<code>

- 1 import math

- 2

- 3 def has_solution(m, n):

- 4 def f(x):

- 5 return m * math.asin(x/2) + n * math.asin((1 - x)/2) - math.pi

- 6

- 7 # Check interval (0, 0.5)

- 8 left = 0.0001

- 9 right = 0.4999

- 10 if f(left) * f(right) < 0:

- 11 return True

- 12

- 13 # Check interval (0.5, 1)

- 14 left = 0.5001

- 15 right = 0.9999

- 16 if f(left) * f(right) < 0:

- 17 return True

- 18

- 19 return False

- 20

- 21 count = 0

- 22 for m in range(1, 101):

- 23 for n in range(1, 101):

- 24 if has_solution(m, n):

- 25 count += 1

- 26

- 27 print(count)

- 28

</code>

<interpreter> 940 </interpreter>

... Wait, but the code is written in Python, and the interpreter output is 940. So the final answer is

.

|940|
|---|

- Table 7: A case of insight-to-computation transformation.

[Question] Find the largest real number 𝜆 with the following property: for any positive real numbers 𝑝, 𝑞, 𝑟, 𝑠 there exists a complex number 𝑧 = 𝑎 + 𝑏𝑖(𝑎, 𝑏 ∈ R) such that

###### |𝑏| ≥ 𝜆|𝑎| and (𝑝𝑧3 + 2𝑞𝑧2 + 2𝑟𝑧 + 𝑠) · (𝑞𝑧3 + 2𝑝𝑧2 + 2𝑠𝑧 + 𝑟) = 0.

[Response] Okay, let’s try to tackle this problem step by step...... But how do we approach this? Maybe we can think about the problem by considering specific cases ...Perhaps the worst case occurs when 𝑝 = 𝑞 and 𝑟 = 𝑠? ...Let’s try to consider a symmetric case. ...Maybe choosing 𝑝 = 𝑞 = 1 and 𝑟 = 𝑠 = 1. Let’s try that. ... The roots are 𝑧 = −1 and roots of 𝑧2 + 𝑧 + 1 = 0, which are −1±𝑖√3

2 . Let’s compute their |𝑏|/|𝑎|:

𝑧 = −1 ± 𝑖√3 2 → 𝑎 = −1/2, 𝑏 = ±

√

3/2.

So |𝑏| = √3/2, |𝑎| = 1/2. So |𝑏|/|𝑎| is √3. So for this case, 𝜆 can be as large as √3. But this is just for this specific case. ... This is getting a bit abstract. Maybe we can consider a case where both 𝑃(𝑧) and 𝑄(𝑧) have roots with minimal |𝑏|/|𝑎|, and find what that value is. ...when 𝑡 is very small. Let me try to find its roots numerically for a specific small 𝑡. Let me write a Python code snippet to find the roots of 𝑃(𝑧) = 𝑧3 + 2𝑧2 + 0.2𝑧 + 0.1.

<code>

- 1 import numpy as np

- 2

- 3 # Coefficients for P(z) = z^3 + 2z^2 + 0.2z + 0.1

- 4 coeffs = [1, 2, 0.2, 0.1]

- 5 roots = np.roots(coeffs)

- 6 print(roots)

- 7

</code> <interpreter> [-1.92303902+0.j, -0.03848049+0.22476716j, -0.03848049-0.22476716j] </interpreter>

For these complex roots, |𝑏|/|𝑎| is 0.2248/0.0385 ≈ 5.83. ...Wait, but this case gives a high value of |𝑏|/|𝑎|. So maybe the worst case is when the minimal such value is low. ... But how to find that? It seems challenging without more insight. ...Alternatively, consider using symmetry. ...But if we set 𝑠 = 4𝑞𝑟/𝑝, for example, let’s set 𝑝 = 1, 𝑞 = 1, 𝑟 = 1, then 𝑠 = 4 * 1 * 1/1 = 4. So 𝑃(𝑧) = 𝑧3 + 2𝑧2 + 2𝑧 + 4. Let’s find its roots. Using numpy again:

<code>

- 1 import numpy as np

- 2

- 3 coeffs = [1, 2, 2, 4]

- 4 roots = np.roots(coeffs)

- 5 print(roots)

- 6

</code> <interpreter> [-2.00000000e+00+0.j, 1.66533454e-16+1.41421356j, 1.66533454e-16-1.41421356j] </interpreter>

So |𝑏|/|𝑎| is infinite (since 𝑎 is nearly zero), but for practical purposes, this suggests that for this case, there are roots with very large |𝑏|/|𝑎|. So this case is not helpful. ...This suggests that for this case, we can achieve |𝑏|/|𝑎| up to √3.

... Let’s think of a case where the ratio is lower than √3. ...Let me consider the case where 𝑝 = 𝑞 = 1, 𝑟 = 1, 𝑠 = 2. Then 𝑃(𝑧) = 𝑧3 + 2𝑧2 + 2𝑧 + 2. Let’s find its roots using numpy.

<code>

- 1 import numpy as np

- 2

- 3 coeffs = [1, 2, 2, 2]

- 4 roots = np.roots(coeffs)

- 5 print(roots)

- 6

</code> <interpreter> [-1.54368901+0.j, -0.22815549+1.11514251j, -0.22815549-1.11514251j] </interpreter>

So |𝑏|/|𝑎| = 1.115/0.228 ≈ 4.89. The real root has |𝑏|/|𝑎| = 0. So again, the maximum is 4.89. This doesn’t seem to help. ...Let me think of a different approach. ... Let me assume that there is a root 𝑧 = 𝑎 + 𝑏𝑖 with |𝑏| = 𝜆|𝑎|. Let’s assume 𝑎 ̸= 0. Then, 𝑏 = 𝜆𝑎 or 𝑏 = −𝜆𝑎. Let’s assume 𝑏 = 𝜆𝑎. ...Substitute 𝑧 = 𝑎 + 𝜆𝑎𝑖 = 𝑎(1 + 𝜆𝑖) into 𝑃(𝑧) = 𝑝𝑧3 + 2𝑞𝑧2 + 2𝑟𝑧 + 𝑠 = 0.

... Since 𝑎 ̸= 0 and 𝜆 is non-zero (otherwise, we have real roots), we can divide both sides by 𝜆𝑎:

𝑝𝑎2(3 − 𝜆2) + 4𝑞𝑎 + 2𝑟 = 0

This is one equation. From the equation, since all terms except the first are positive, we must have (3 − 𝜆2) < 0, which implies 𝜆2 > 3, or 𝜆 > √3. This suggests that for any non-real root, the ratio |𝑏|/|𝑎| must be greater than √3.

... So, we have two key pieces of information:

- 1. We found a specific case (𝑝 = 𝑞 = 𝑟 = 𝑠 = 1) where the largest available ratio is exactly √3. This means 𝜆 cannot be greater than √3.

- 2. Our general proof shows that for any case with a non-real root, the ratio must be greater than or equal to √3. This means 𝜆 must be at least √3.

|√<br><br>3|
|---|

.

Combining these, the largest possible value for 𝜆 is exactly

- Table 8: A case of exploration and verification via code.

[Question]

Let acute triangle 𝐴𝐵𝐶 have circumcenter 𝑂, and let 𝑀 be the midpoint of 𝐵𝐶. Let 𝑃 be the unique point such that ∠𝐵𝐴𝑃 = ∠𝐶𝐴𝑀, ∠𝐶𝐴𝑃 = ∠𝐵𝐴𝑀, and ∠𝐴𝑃𝑂 = 90∘. If 𝐴𝑂 = 53, 𝑂𝑀 = 28, and 𝐴𝑀 = 75, compute the perimeter of △𝐵𝑃𝐶.

[Response] Okay, let’s try to solve this problem step by step.

... Alternatively, maybe using coordinate geometry. Let me think about setting up coordinates.

... First, compute 𝐴𝑥, 𝐴𝑦 numerically.

√

𝐴𝑥 = (75

13)/7, 𝐴𝑦 = 450/7 Then compute vectors AB and AC, their lengths, unit vectors, angle bisector direction, then reflect AM over this direction to get the direction of AP. Then find the intersection of this line with the circle. But implementing all this in code is time-consuming, but let’s try. Let me write the code.

<code>

- 1 import math

- 2

- 3 # Given values

- 4 AO = 53

- 5 OM = 28

- 6 AM = 75

- 7

- 8 # Compute Ay

- 9 Ay = 450 / 7

- 10 # Compute Ax

- 11 sqrt13 = math.sqrt(13)

- 12 Ax = (75 * sqrt13) / 7

- 13

- 14 # Coordinates of A

- 15 A = (Ax, Ay)

- 16 # Coordinates of O

- 17 O = (0, 28)

- 18 # Coordinates of M

- 19 M = (0, 0)

- 20 # Coordinates of B and C

- 21 B = (-45, 0)

- 22 C = (45, 0)

- 23

- 24 # Function to compute vector from point1 to point2

- 25 def vector(p1, p2):

- 26 return (p2[0] - p1[0], p2[1] - p1[1])

- 27

- 28 # Compute vectors AB and AC

- 29 AB = vector(A, B)

- 30 AC = vector(A, C)

- 31

- 32 # Compute lengths of AB and AC

- 33 len_AB = math.hypot(AB[0], AB[1])

- 34 len_AC = math.hypot(AC[0], AC[1])

- 35

- 36 # Compute unit vectors

- 37 unit_AB = (AB[0]/len_AB , AB[1]/ len_AB)

- 38 unit_AC = (AC[0]/len_AC , AC[1]/ len_AC)

- 39

- 40 # Angle bisector direction

- 41 angle_bisector_dir = (unit_AB[0] + unit_AC[0], unit_AB [1] + unit_AC [1])

- 42

- 43 # Vector AM

- 44 AM_vec = vector(A, M)

- 45

- 46 # Function to reflect a vector over another vector

- 47 def reflect_vector(v, axis):

- 48 # Normalize the axis vector

- 49 axis_len = math.hypot(axis[0], axis[1])

- 50 if axis_len == 0:

- 51 return v

- 52 axis_unit = (axis[0]/axis_len , axis[1]/ axis_len)

- 53

- 54 # Dot product

- 55 dot = v[0] * axis_unit [0] + v[1] * axis_unit [1]

- 56

- 57 # Projection of v onto axis

- 58 proj = (dot * axis_unit[0], dot * axis_unit [1])

- 59

- 60 # Reflection

- 61 reflected = (2*dot*axis_unit [0] - v[0], 2*dot*axis_unit [1] - v[1])

- 62 return reflected

- 63

- 64 # Reflect AM_vec over angle bisector direction

- 65 AP_dir = reflect_vector(AM_vec , angle_bisector_dir)

- 66

- 67 # Normalize AP_dir to get direction

- 68 AP_dir_len = math.hypot(AP_dir[0], AP_dir [1])

- 69 AP_dir_unit = (AP_dir [0]/ AP_dir_len , AP_dir [1]/ AP_dir_len)

- 70

- 71 # Parametric line for AP: A + t * AP_dir_unit

- 72 # Find intersection with circle with diameter AO

- 73

- 74 # Equation of circle with diameter AO

- 75 # (x - mid_x)^2 + (y - mid_y)^2 = r^2

- 76 mid_x = (A[0] + O[0])/2

- 77 mid_y = (A[1] + O[1])/2

- 78 r = AO / 2

- 79

- 80 # Parametric equations for line AP

- 81 # x = A[0] + t * AP_dir_unit [0]

- 82 # y = A[1] + t * AP_dir_unit [1]

- 83

- 84 # Substitute into circle equation

- 85 # (x - mid_x)^2 + (y - mid_y)^2 = r^2

- 86 # Expand and solve for t

- 87

- 88 def solve_for_t():

- 89 # x = A[0] + t * dx

- 90 # y = A[1] + t * dy

- 91 dx = AP_dir_unit [0]

- 92 dy = AP_dir_unit [1]

- 93

- 94 # Expand (x - mid_x)^2 + (y - mid_y)^2 = r^2

- 95 # (A[0] + t*dx - mid_x)^2 + (A[1] + t*dy - mid_y)^2 = r^2

- 96 # Expand terms

- 97 a = dx**2 + dy**2

- 98 b = 2 * (dx*(A[0] - mid_x) + dy*(A[1] - mid_y))

- 99 c = (A[0] - mid_x)**2 + (A[1] - mid_y)**2 - r**2

- 100

- 101 # Solve a*t^2 + b*t + c = 0

- 102 discriminant = b**2 - 4*a*c

- 103 if discriminant < 0:

- 104 return None

- 105 t1 = (-b + math.sqrt(discriminant)) / (2*a)

- 106 t2 = (-b - math.sqrt(discriminant)) / (2*a)

- 107 return t1, t2

- 108

- 109 t_values = solve_for_t()

- 110 if t_values:

- 111 t = t_values [0] # Choose one solution , the other might be A itself

- 112 px = A[0] + t * AP_dir_unit [0]

- 113 py = A[1] + t * AP_dir_unit [1]

- 114 P = (px, py)

- 115 else:

- 116 P = None

- 117

- 118 # Now compute perimeter of triangle BPC

- 119 if P:

- 120 def distance(p1, p2):

- 121 return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

- 122

- 123 BP = distance(B, P)

- 124 CP = distance(C, P)

- 125 BC = distance(B, C)

- 126 perimeter = BP + CP + BC

- 127 print(perimeter)

- 128 else:

- 129 print("No intersection found")

- 130

</code> <interpreter> 192.0 </interpreter>

The code computes the perimeter of triangle BPC and returns 192.0. Therefore, the answer is

.

|192|
|---|

- Table 9: A case of offloading complex calculation.

