## Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### Xinyu Yang∗†, Yuwei An∗†, Hongyi Liu†, Tianqi Chen†‡, Beidi Chen† †Carnegie Mellon University, ‡Nvidia

Autoregressive Large Language Models (AR-LLMs) frequently exhibit implicit parallelism in sequential generation. Inspired by this, we introduce Multiverse, a new generative model that enables natively parallel generation. Multiverse internalizes a MapReduce paradigm, generating automatically through three stages: (i) a Map stage for adaptive task decomposition, (ii) a Process stage for parallel subtask execution, and (iii) a Reduce stage for lossless result synthesis. Next, we build a real-world Multiverse reasoning model with co-design of data, algorithm, and system, enabling rapid and seamless transfer from frontier AR-LLMs. For data creation, we develop Multiverse Curator, an automated LLM-assisted pipeline that transforms sequential reasoning chains into structured training data, avoiding costly human annotations. Algorithmically, we design Multiverse Attention to separate parallel reasoning steps while keeping compatibility with causal attention for efficient training. Systematically, we implement Multiverse Engine to support parallel inference. It features a dedicated interpreter that dynamically switches between sequential and parallel generation, triggered directly by the model. After a 3-hour finetuning with 1K examples, our Multiverse-32B stands as the only open-sourced non-AR model achieving performance on par with leading AR-LLMs of the same scale, evidenced by AIME24 & 25 scores of 54% and 46%, respectively. Moreover, our budget control experiments show that Multiverse-32B exhibits superior scaling, outperforming AR-LLMs by 1.87% on average using the same context length. Such scaling further leads to practical efficiency gain, achieving up to 2× speedup across varying batch sizes. We have open-sourced the entire Multiverse ecosystem, including data, model weights, engine, supporting tools, as well as complete data curation prompts and detailed training and evaluation recipes.

arXiv:2506.09991v2[cs.LG]13Jun2025

##### Comparison among various generative modeling methods

Blocks executed in serial

Blocks executed in parallel

Map phase adaptively split tasks

Reduce phase losslessly merge results

Generation following sequential logic

[Figure 10]

Github: https://github.com/Multiverse4FM/Multiverse Website: https://Multiverse4FM.github.io

[Figure 11]

###### Autoregressive Model Diffusion Model

Multiverse Model

Analysis: prove an equation

Aha!

Analysis

[Mask] [Mask] … [Mask] [Mask]

Split

[Mask] Subtask 1 … Wait [Mask]

Subtask 1

Subtask 2

Subtask 2: simplify RHS

Subtask 1: simplify LHS

Wait, wait.

Hmm.

Analysis Subtask 1 … Wait [Mask]

Merge

Verification

Conclusion

Analysis Subtask 1 … Wait, wait.

Conclusion: LHS = RHS

Sequential blocks Parallel blocks Map phase: adaptive task split Reduce phase: lossless result merge

- Figure 1 Model Overview. Autoregressive models are limited to sequential generation, while diffusion models ignore logical dependency in parallel generation. In contrast, Multiverse models enable adaptive and lossless parallel generation.

###### 1 Introduction

“In an infinite multiverse, everything that can happen does happen—somewhere.”

Test-time scaling has advanced Large Language Models (LLMs) by increasing the generation length (Jaech et al., 2024; Guo et al., 2025) and depth (Geiping et al., 2025), closely reflecting human cognition. However, empowered by modern hardware like GPUs, ideal LLMs can surpass humans by scaling a third dimension: width, which allows parallel task-solving. Realizing this potential requires LLMs to “smartly” parallelize and merge their generation, following the classic MapReduce paradigm (Dean et al., 2004): splitting into subtasks, processing them independently in parallel, and merging their results. Such philosophy has a long history in computer science (McCarthy, 1960; Aho et al., 1974) while driving fundamental progress in other fields including manufacturing (Hounshell, 1984), agriculture (Netting, 1993), and finance (Dean and Ghemawat, 2008). This shift from sequential to parallel task-solving unlocks economies of scale: reducing the time per unit and keeping near-constant overall latency as task complexity grows, thereby offering a promising path towards artificial superintelligence (ASI).

Despite this potential, current LLMs are limited by the inherently sequential nature of autoregressive (AR) generation. While non-AR architectures, such as diffusion models (Sahoo et al., 2024; Zhao et al., 2025) and consistency models (Kou et al., 2024), along with their hybrid semi-AR variants (Arriola et al., 2025; Nie et al., 2025), natively enable parallel generation, they incur substantial computational waste. Their rigid, brute-force parallelism ignores logical dependencies, partly due to a lack of real-world training data to supervise when and how parallel generation should occur. Another stream of research (Zhang et al., 2024; Cobbe et al., 2021; Yao et al., 2023; Pan et al., 2025) leverages external tools to parallelize or merge tasks heuristically, leading to the loss of internal states, like the intermediate reasoning steps, during communication with external modules. Although our concurrent work (Jin et al., 2025; Rodionov et al., 2025) allows internal communication, they introduce inconsistencies between training and inference, limiting their effectiveness to short sequences with shallow parallelism. These challenges raise a research question: How to design a modeling framework for LLMs that can (i) adaptively split and merge tasks, (ii) losslessly preserve internal states, and (iii) generally apply to diverse parallelism patterns?

Due to the dominance of AR-LLMs, we start to answer it by revealing numerous intrinsic parallelism in the sequential outputs of these models. Specifically, we analyze the long Chain-of-Thought (CoT) trajectories from the s1K-1.1 dataset (Muennighoff et al., 2025). Among them, over 98% exhibit parallelizable branches, despite being trained only for sequential generation. These branches, as shown in Figure 2, fall into collective and selective ones that appear frequently within individual CoT trajectories, either consecutively or recursively, covering a wide range of scenarios. However, our prompting and probing tests verify that AR-LLMs cannot actively enforce or discern such parallelism. These findings motivate the design of a new modeling framework that can be bootstrapped directly from pre-trained AR-LLMs, which further requires us to address three practical limitations: (i) Data: Real-world CoT trajectories lack explicit parallelism. (ii) Algorithm: Transformers with causal attention are limited to sequential generation. (iii) System: Inference engines for AR-LLMs cannot execute parallel generation.

To achieve these, we introduce Multiverse, a generative modeling framework built on the MapReduce paradigm that dynamically adjusts its parallelism during generation. It internalizes a three-stage pipeline: a sequential Map stage performs adaptive task decomposition; a parallel Process stage allows independent subtask execution; and a sequential Reduce stage ensures lossless result synthesis. Moreover, the pipeline can invoke itself recursively, enabling optimal time complexity with unlimited resources. We theoretically prove this optimality on a synthetic NP-hard SAT problem, demonstrating that Multiverse is the only framework that achieves a linear-time solution. Building on this, we co-design our data, algorithm, and system, providing a universal approach to building a realworld Multiverse model for complex reasoning tasks, offering a seamless and rapid transition from AR-LLMs.

Data Curation. In Section 5.1, we develop Multiverse Curator, an automated LLM-assisted pipeline that transforms sequential reasoning chains into parallel structures via five steps: (i) parsing the sequential chain into a summary tree; (ii) identifying parallelizable nodes within the summary tree; (iii) reformatting the summary into a parallel generation structure; (iv) refilling original reasoning steps into this structure; and (v) adding Map & Reduce stages while rewriting Process stage. Moreover, content and grammar checks are performed to flag lowquality data for regeneration, avoiding costly manual filtration and annotation. In practice, this process results in Multiverse-1K, a dataset of 1,000 high-quality structured training samples for advancing LLM reasoning.

Algorithm Design. In Section 5.2, we design Multiverse Attention to enable parallel generation while keeping training efficiency. This is achieved by modifying attention masks and position embeddings to strictly

separate independent reasoning branches in attention calculation, which can be trained in parallel, similar to causal attention. This design also excels in data efficiency: since these changes are minor, pre-trained AR models can be rapidly transferred from causal attention to Multiverse attention using only a few thousand examples.

System Implementation. In Section 5.3, we implement Multiverse Engine featuring a specialized interpreter to support MapReduce execution. By interpreting control tags generated by Multiverse models, our engine can dynamically switch between sequential and parallel generation without overhead, yielding a flexible workflow. This includes (i) Sequential → Parallel: mapping subtasks to separate branches for parallel execution with prefix sharing, and (ii) Parallel → Sequential: reducing Key-Value (KV) states from all branches back into one sequence.

The integration of these modules enables efficient training and inference of Multiverse models. Specifically, we develop Multiverse-32B by applying supervised fine-tuning (SFT) to Qwen-2.5-32B-Instruct using only 1K examples within 3 hours. Empirically, Multiverse-32B achieves significant improvements in reasoning ability, outperforming the base model by 23.6%, with AIME24 and AIME25 scores of 53.8% and 45.8%, respectively. These results are comparable to AR-LLMs, confirming that Multiverse does not compromise model performance. Furthermore, Multiverse-32B exhibits more efficient test-time scaling, yielding an average improvement of 1.87% within fixed latency constraints. This efficiency stems from its parallel generation capabilities, leading to up to 2× wall-clock speedup per generated token while maintaining effective scaling across variable batch sizes range from 1 to 128.

We have open-sourced the entire Multiverse ecosystem, including data, model weights, engine, and supporting tools, along with complete data curation prompts and detailed training and evaluation recipes. We hope this fullstack release will inspire and accelerate advancements in developing more efficient and scalable generative models.

###### 2 Related Work

Test-time Scaling. Prior work has shown that optimizing AR-LLMs to generate longer outputs improves their reasoning abilities. This is evident in frontier reasoning models built with reinforcement learning (RL) (OpenAI

- et al., 2024; DeepSeek-AI et al., 2025; OpenAI, 2025; Google, 2025b; xAI, 2025), and also validated through supervised fine-tuning (SFT) on smaller models with a few distilled examples (Muennighoff et al., 2025; Ye et al., 2025b). However, this length scaling greatly increases latency due to the sequential nature of AR generation. Other methods like depth scaling (Geiping et al., 2025; Zhao et al., 2025) suffer from the same issue, while width scaling (Brown et al., 2024; Pan et al., 2025) requires external tools/models to split or merge generations.

Internal Parallel Generation. Recent work has increasingly explored other models to replace the commonly used AR models, thereby enabling parallel generation. Among them, discrete diffusion models (Sahoo et al., 2024; Shi et al., 2024; Austin et al., 2021; Lou et al., 2023; Wang et al., 2025), including masked and absorbed variants, are gaining growing attention. To narrow their gap with AR models, efforts have been made on methods like hybrid AR-diffusion generation (Arriola et al., 2025; Fathi et al., 2025) and training/test-time scaling (Nie

- et al., 2025; Zhao et al., 2025; Ye et al., 2025a). However, (Feng et al., 2025) has theoretically shown that these approaches cannot reduce the number of sequential generating or sampling steps, as they brute-force parallelize token generation without adhering to inherent relations. Similarly, other work explores continuous diffusion models (Barrault et al., 2024) and consistency models (Kou et al., 2024). Among these open-sourced, non-AR models, a common issue is their current inability to scale to complex reasoning tasks, such as AIME (Mathematical Association of America, 2024). While our concurrent work (Jin et al., 2025; Rodionov et al., 2025) begins to explore the use of customized attention masks for parallel generation, their design are not general or adaptive, limiting their effectiveness to shallow, non-nested parallelism. In contrast, our Multiverse offers a more efficient and scalable approach to enable internal parallel generation, which is generally applicable to diverse parallelism patterns.

External Parallel Generation. In another line of research, several approaches leverage external tools or models to enable parallel generation (Yao et al., 2023; Pan et al., 2025; Wang et al., 2022; Brown et al., 2024; Zhang et al., 2024). However, these methods generally leverage heuristic rules and external tools to parallelize or merge their generation. For instance, Best-of-N (Brown et al., 2024) and self-consistency (Wang et al., 2022) use a brute-force approach by parallelizing generation at the beginning of generation. Other methods like Monte Carlo tree search (MCTS) (Zhang et al., 2024) and Tree of Thoughts (ToT) (Yao et al., 2023) offer more fine-grained parallelism, yet they are still fundamentally guided by heuristics and depend on an external verifier. While recent work (Pan et al., 2025) enables more adaptive parallel generation, it suffers from significant information loss when

Model Collaboration (External)

Analysis

Analysis

Fork

Analysis

parallelizing and merging branches, as it requires inter-model communication when switching between sequential and parallel generation, during which short text summaries rather than complete KV states can be shared.

Subtask 1

Subtask 2

Path 1 Path 2

Path 1

Path 2

###### 3 Long CoT Generation: Sequential or Parallel in Logic?

Answer 1 Answer 2

External Model Verification

External Model Execution

External Model Verification

Path 1.1 Path 1.2

Summary 1 Summary 2

This section presents several key observations of parallelism potential in AR-LLMs. First, in Section 3.1, we examine the CoT outputs from such models, verifying the common existence of intrinsic parallelism. Next, Section 3.2 details two tests, showing that AR-LLMs cannot explicitly enforce or discern this parallelism during generation.

[Figure 12]

[Figure 13]

Concatenated Summary

Score=0.8 Score=0.6

External Model Verification

Join

Conclusion

Conclusion

Conclusion

###### 3.1 LLMs can Implicitly Generate Parallelizable Branches.

(b). MCTS selects only one branch and loses information from all others.

(c). APR loses context when utilizing external models for parallel execution.

(a). Best-of-N utilize external models to select branches.

We start by analyzing the long CoT trajectories of AR-LLMs using the s1K-1.1 dataset (Muennighoff et al., 2025), including both Deepseek R1 (Guo et al., 2025) and Gemini 2.0 Flash Thinking (Google, 2024), aiming to answer the research question: Does the logic of sequentially generated tokens truly rely on all context that precedes it?

Multiverse Model (Internal)

Calculate GCD(96, 160)

- Test a from 1 to 3

- a=1 a=2 a=3

Split

Merge Test b from 1 to 3

- b=1 b=2 b=3

Analysis the problem

Split

Split

Path 1 Path 2

Path 1: Euclidean algorithm

Path 2: Prime Factorization

Split

k=2 k=3

k=1 Split

k=i+1

Result=32 Result=45

| | |
|---|---|
|Mer|ge|

[Figure 14]

[Figure 15]

Verification Verification

Sub-conclusion 1 Sub-conclusion 2

Merge

Merge

Merge

The correct answer for GCD(96, 160) is 32.

Conclusion

Conclusion

(a). Path exploration and verification using two selective branches alone.

(b). Case study using collective branches in a consecutive structure.

(c). Case study when exploring one path within a recursive structure.

- Figure 2 Existence of Implicit Parallelism. Parallelizable branches fall into collective (with all branches contributing) and selective (with some branches contributing) ones. They occur alone or as a part in consecutive or recursive structures.

Surprisingly, we reveal instances of “parallelizable branches”, where multiple independent logic steps can execute concurrently, rather than strictly sequentially. These branches highlight the inherent parallelism in AR-LLMs.

- Figure 2 classifies them as collective or selective, which appearing flexibly in consecutive and recursive structures.

Collective Branches involve splitting a task into multiple subtasks can be processed concurrently, whose outputs are merged into a final result. Examples include studying various cases and analyzing individual problems.

Selective Branches refer to scenarios where numerous potential paths are considered, but not all necessarily contribute to the final output. Examples include exploring diverse solutions or examining competing hypotheses.

Table 1 Statistics of Implicit Parallelism. Parallelizable branches commonly exist in long CoT trajectories generated by AR-LLMs. Per-example existence ratio (R%) and frequency (F) of different types are measured in the format R|F.

###### Collective Branch Selective Branch

Total Case Study Subtask Execution Other Path Exploration Other

Deepseek R1 28.0 | 2.00 47.3 | 3.38 3.2 | 0.23 19.4 | 1.39 1.1 | 0.07 99.0 | 7.07 Gemini 2.0 Flash 39.4 | 2.82 45.0 | 3.22 2.9 | 0.21 8.3 | 0.60 1.4 | 0.09 97.0 | 6.94

Table 1 further details the occurrence ratios and frequencies for various types within the s1K-1.1 data. Notably, over 98% of examples feature parallelizable branches. Among them, collective branches, such as case studies and

subtask executions, are predominant, accounting for 79%, with selective branches like path exploration comprising the remaining 19%. Moreover, these branches occur frequently, appearing an average of 7 times each example.

- 3.2 LLMs cannot Explicitly Structure Parallelizable Branches. Next,we show that AR-LLMs cannot explicitly enforce or discern such parallelism from token and hidden spaces.

Deepseek R1

60

100

20

57%

55

53%

###### Percentage(%)

50

10

###### Accuracy(%)

50

48%

0

0

0 1 2

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15+

45

Gemini 2.5 Pro

100

20

40

50

10

35

0

0

0 1 2

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15+

30

QWQ-32B R1-32B R1-70B

# Explicit

# Implicit

(a) Prompting Test: Comparing Explicit and Implicit Structure Counts

(b) Probing Test: Classifier Accuracy

- Figure 3 Our prompting and probing tests show AR-LLMs cannot explicitly enforce or discern parallelizable branches.

Prompting Test. We first prompt Deepseek R1 and Gemini 2.5 Pro using the same questions, with a detailed description of all valid parallel structures. Figure 3a showcases a significant 90% disparity emerges between explicit occurrence and implicit existence of these structures, indicating that AR-LLMs struggle to generate in parallel.

Probing Test. We then probe into the hidden space of AR-LLMs, confirming whether they can discern intrinsic parallelism. Specifically, we label tokens before parallel blocks as positive examples and treat all others as negative. Final-layer representations of these tokens are extracted using DeepSeek-R1-Distill-Qwen-32B & 70B (DeepSeek-AI et al., 2025) and QWQ-32B (Qwen, 2025). A two-layer MLP classifier is trained to predict whether a token initiates parallelizable branches. However, the classifier’s performance, which was comparable to random guessing (as shown in Figure 3b), suggests that AR-LLMs do not truly understand such parallelism. Instead, they generate these structures unconsciously, based on patterns learned from their pre-training corpus.

- 4 Designing Multiverse for Natively Parallel Generative Modeling.

With all findings in Section 3, we present Multiverse, a novel generative modeling framework built on the MapReduce paradigm, which adaptively parallelizes and losslessly merges its generation to surpass AR models.

###### 4.1 Preliminaries.

Language Modeling aims to learn the joint probability distribution P(x1,x2,...,xL) over sequences of words or tokens given a finite vocabulary V of tokens, and a sequence of L tokens denoted by x1:L = (x1,x2,...,xL).

Autoregressive Modeling represents sequence x1:L from left to right, where each token xt is conditioned on all past tokens x1:t−1. Thus, the joint probability of x1:L is factorized as a product of conditional probabilities:

P(x1:L|θAR) = P(x1,x2,...,xL|θAR) =

L

P(xt|x1,...,xt−1;θAR)

t=1

where θAR denotes model parameters. AR models offer high accuracy but exhibit poor parallelism in generation.

### Comparison among various generative modeling methods

Blocks executed in serial

Blocks executed in parallel

Map phase adaptively split tasks

Reduce phase losslessly merge results

Generation following sequential logic

#### Autoregressive Model Diffusion Model

Multiverse Model

Analysis: prove an equation

Aha!

Analysis

[Mask] [Mask] … [Mask] [Mask]

Split

[Mask] Subtask 1 … Wait [Mask]

Subtask 1

Subtask 2

Subtask 2: simplify RHS

Subtask 1: simplify LHS

Wait, wait.

Hmm.

Analysis Subtask 1 … Wait [Mask]

Merge

Verification

Conclusion

Analysis Subtask 1 … Wait, wait.

Conclusion: LHS = RHS

Serial blocks Parallel blocks Map phase: adaptive task split Reduce phase: lossless result merge

x1

x2 x1,1 x1,2 x1,3 x1,4 x1,5 x2,1 x2,2 x2,3 x2,4 x2,5 x3 x4

- x1,s x1,1 x1,2 x1,3 x1,4 x1,5 x1,e
- x2,s x2,1 x2,2 x2,3 x2,4 x2,5 x2,e

x1 x2 xs

xe x3 x4

Map Stage Process Stage (execution in parallel) Reduce Stage

Figure 4 Multiverse enables adaptive and lossless parallel generation by natively internalizing a MapReduce paradigm.

###### 4.2 Multiverse Modeling.

Our modeling framework, Multiverse, advances beyond AR by eliminating redundant sequential dependencies between independent contexts, allowing for adaptive and lossless control over the start and end of parallel generation. To realize this, we adopt a MapReduce structure internalizing three stages, as illustrated in Figure 4.

Map Stage. The pipeline begins by sequentially generating a concise task decomposition plan, denoted as xs. Each subtask in xs is then mapped to an independent prefix sequence, modeled as P(x1,s|xs) and P(x2,s|xs).

Process Stage. Next, it performs parallel modeling for each branch independently, conditioned on its own prefix. This enables the concurrent generation of diverse branches, like: P(x1,1:6|x[1:3,s],x1,s) and P(x2,1:6|x2,s,x[1:3,s]). Each branch ends when a specific suffix (i.e, x1,e or x2,e) is generated. We use the same suffix for all branches. Reduce Stage. After completing all branches, Multiverse shift back to sequential generation to conclude them, which is conditioned on preceding tokens from all branches, modeled as P(xe,[3:4]|x1,[s,1:6,e],x2,[s,1:6,e],x[1:3,s]). This structure enables Multiverse to: (i) adaptively decide how to parallelize generation during the Map stage; and (ii) retain information by ensuring every branch remains fully accessible in the Reduce stage and beyond. Notably, Multiverse generalizes to both recursive and consecutive compositions of multiple MapReduce structures.

###### 4.3 Structured Generation Flow.

To enable automatic control over the generation flow, Multiverse further employs a structured set of specialized control tags that explicitly define each MapReduce block. These tags, such as <Parallel> and <Path>, delineate the boundaries of MapReduce blocks and coordinate the execution of all three internal stages. Figure 5 provides an example of this structure.

Map Phase

… Let’s think in parallel! <Parallel> <Goal> Consider two cases.

- <Outline> 1.failure condition sin2x + sinxcosx ≤ cos2x </Outline>
- <Outline> 2.failure condition cos2x + sinxcosx ≤ sin2x </Outline> </Goal>

Process Phase

<Path> 1. Let's rewrite this as: sin2x cos2x + sinxcosx ≤ 0 Hmm, perhaps factoring? … Let me denote t = tanx, then we have t2 + t <=1 … Therefore, the inequality sin2x + sinxcosx <= cos2x holds when x is in (0°, alpha], where alpha=arctan([sqrt5 -1]/2). </Path>

<Path> 2. Let's handle the other inequality: cos2x + sinxcosx <= sin2x. … Perhaps use division by cos2x (since cosx >0). Divide by cos2x:1 - tan2x + tanx <=0 … Let that angle be beta = arctan([1 + sqrt5]/2 )≈58.3°.

The MapReduce block begins with the <Parallel> tag, initiating the three-stage process. Immediately after it, the Map stage starts with the <Goal> tag, which defines the overall objective, which is then broken down into subtasks using multiple nested and indexed <Outline> tags. Following goal specification (signaled by </Goal>), the Process stage commences. At this stage, each subtask is independently mapped and processed within a <Path> block in parallel, matched by its index. Once all paths have finished (signaled by </Path>), the <Conclusion> tag triggers the Reduce stage that synthesizes the results from these independent paths into a final coherent output, which is ended with the </Conclusion> tag. Finally, the MapReduce block is terminated by the </Parallel> tag.

x is in [ beta, 90°). </Path>

Reduce Phase

<Conclusion> The two inequalities are each responsible for an interval: (0°, alpha] for the first inequality and [beta, 90° ) for the second. Note that [sqrt5 -1]/2 multiplied by [1 + sqrt5]/2 = (5 -1)/4 = 4/4=1.

This is useful because it shows that alpha and beta are complementary angles in the sense that tan alpha * tan beta=1, so tan beta=1/tan alpha, which implies that beta=90° - alpha. </Conclusion> </Parallel>

Figure 5 Example of MapReduce Structure.

###### 5 Building a Real-world Multiverse Model.

To deploy Multiverse in real-world scenarios, we present a comprehensive suite consisting of Multiverse Curator as the data generator, Multiverse Attention as the core algorithm, and Multiverse Engine as the optimized system.

This suite enables a smooth and rapid shift from leading AR models to Multiverse models. In particular, we apply this suite in complex reasoning tasks, leading to a Multiverse model that exhibits strong reasoning capabilities.

Q: Given a rational number, write it as a fraction in lowest terms and calculate the product of the resulting numerator 𝑎 and denominator 𝑏. For how many rational numbers between 0 and 1 will 20! = 𝑎 × 𝑏 be the resulting product?

|Inherent Parallelism in Original Reasoning Chain|
|---|

… To compute the exponents of primes in 20!: For prime 2: floor(20/2) + floor(20/4) + floor(20/8) + floor(20/16) =10+5+2+1=18; For prime 3: floor(20/3) + floor(20/9) + floor(20/27)=6 +2 +0=8; For prime 5: floor(20/5) + floor(20/25)=4 +0=4; Prime 7: floor(20/7) + floor(20/49)=2 +0=2 …

|1. Parse the Chain into a Tree-structure Summary|
|---|

Step 1 Step 2.1

Step 2 Step 2.2 Step 2.3 Step 3 Step 4 Step 5

|2. Identify Parallel Nodes|
|---|

[Step 2.1-2.2, Step 2.3]: Collective Branches

[Step 4, Step 5]: Selective Branches

- 3. Reformat the Summary Tree into the MapReduce Structure via Control Tags

- 4. Refill the Full, Detailed

- Step 4

- Step 5

Step 2.1-2.2

Step 1-2

<Parallel> </Parallel> Step 3

<Parallel>

</Parallel>

Find</Goal>: Start Parallel Runner, using the shared prefix from sequential Thhread

Find<Goal>: Start

Find all </Path>, stop parallel thread

- Find<Summarization> 1
- Find<Summarization> 2

Step 2.3

…Let’s think in parallel! <Parallel> <Goal> Compute the exponents of primes in 20! <Outline> 1: Prime 2 </Outline> <Outline> 2: Prime 3 </Outline> … </Goal>

Reasoning Trajectories

- <Path> 1: floor(20/2) + floor(20/4) + floor(20/8) + floor(20/16) =10+5+2+1=18 </Path>
- <Path> 2: floor(20/3) + floor(20/9) + floor(20/27)=6 +2 +0=8 </Path> … <Conclusion> The exponents for Prime 2,3,5,7,11 are 18, 8, 4, 2, 1, respectively. </Conclusion> </Parallel> …

|5. Add Map and Reduce Stages, Rewrite All Paths|
|---|

(a) Multiverse Curator automatically generated Multiverse-1K using an LLM-assisted data curation pipeline.

Generator

… </Goal>

- <Path> 1: … </Path>

- <Path> 2: … </Path>

…<Parallel> <Goal>

<Conclusion>… </Conclusion> Let’s move back to sequential mode!

- <Outline> 1: …</Outline>
- <Outline> 2: …</Outline> </Goal>…

<Path>

… </Path>

|1. Generate Structured Reasoning Summary|
|---|

𝑝1, …… , 𝑝

<Path>

Interpreter

… </Path>

[Figure 16]

[Figure 17]

[Figure 18]

|2. Identify Parallel Groups|
|---|

Detecting <Parallel> Detecting <Goal> Starting Map Phase Recording <Outline> 1 and 2

Process Finish Starting Reduce Phase Prefill <Conclusion> Concluding All Paths.

Detecting </Goal>

Map Stage: XXXX XXXX

𝑝1, …… , 𝑝

- Prefill <Path> 1 for Task 1

- Prefill <Path> 2 for Task 2 Starting Parallel Process

<Path>

… </Path>

<Parallel>

𝑝1, …… , 𝑝

<Path>

<Goal> …

###### Memory Pool

… </Path>

[Figure 19]

[Figure 20]

KV Cache Indices Path 1 Path 2 Prefix Decoding Indices

- <Outline> 1: … </Outline>

- <Outline> 2: … </Outline>

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

<Con> …

</Con>

max(𝑝 , 𝑝 , 𝑝 , 𝑝 ) + 1

</Goal>

(b) Multiverse Attention

###### (c) Multiverse Engine

- <Path> 1: … </Path>

- <Path> 2: … </Path>

- Figure 6 Instantiation Overview. Multiverse co-design data (Multiverse Curator), algorithm (Multiverse Attention), and system (Multiverse Engine) to enable real-world reasoning abilities through a rapid and seamless shift from AR-LLMs.

</Conclusion> … </Conclusion>

###### 5.1 Data Curation: Multiverse Curator.

</Parallel>

| |Seq KV1<br><br>ft d|en dr|i es|nh s pi|er th pe|Seq KV2<br><br>en is li|t a ne|ly bs<br><br>t|co en ha|n ce t|ta<br><br>o au|in f<br><br>t|M M om|Path KV1<br><br>a ap<br><br>a|pR R ti|e ed ca|du u lly|c ce|e s s tra|Path KV2<br><br>tr tru n|uc c<br><br>sf|tu<br><br>tu or|r re m|es s s|Path KV3<br><br>, e in se|xp ex qu|li is en|ci ti<br><br>t|tly ng ia|l|Seq KV3| | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

While the long CoT trajectories generated by AR-LLMs generating them is difficult, as detailed in Section 3. To data, we introduce Multiverse Curator, an LLM-assisted reasoning chains into parallel MapReduce structures. This convert is guided by a five-stage prompting protocol powered by Gemini 2.5 Pro (Google, 2025a) in Figure 6a. The detailed prompts are available in Appendix A.

Scheduler

Generating a Summary Tree. First, we iteratively decompose and outline the original reasoning chain into a two-level tree-structured summary. In the first round, the entire reasoning chain is broken down into multiple

7

<Path> For Prime 2 …</Path>

<Path> For Prime 3 …</Path>

steps. In the second round, each step is examined by the LLM, with complex steps being further split into substeps. Finally, every identified step or substep will be clearly labeled and outlined with a concise descriptive summary.

Identifying Parallel Groups. Second, we instruct the LLM to analyze the relationship between consecutive reasoning steps, identifying which steps or step groups can execute in parallel without violating logical dependencies.

Reformating into Parallel Structures. Third, the summary tree is transformed into a parallel structure using the grouping results from the previous step. To signal parallel execution, parallelizable steps or step groups are explicitly marked by enclosing them in the control tags <Parallel> and </Parallel>, forming a parallel block.

Refilling Original Details. Fourth, we prompt the LLM to repopulate the detailed content for each step and substep while keeping the structures. The LLM will retrieve and copy the related texts from original trajectories.

Adding MapReduce Structures & Rewriting All Paths. Finally, we further convert the parallel structures into the MapReduce structures defined in Section 4.3. For each parallel block, the LLM generates both the Map and Reduce stages itself by outlining the specific targets and results for each individual path. Moreover, all paths are rewritten to avoid words implying sequential relations (e.g., “Similarly” or “Alternatively”) and to prevent including or referencing content from other paths, ensuring the completeness and independence of each path. To further enhance our data quality, two supplementary validation stages have been incorporated. After the fourth stage, a content check will filter out data if its edit distance ratio is above 0.2. Next, after the fifth stage, a grammar check will confirm strict adherence to our MapReduce structures. Data failing either case will be iteratively regenerated through our pipeline until both standards are met. We provide more details in Appendix A. The application of this automated pipeline to the s1K-1.1 dataset has yielded Multiverse-1K, a new dataset consisting of 1,000 high-quality, structured reasoning trajectories across a range of math and science problems.

###### 5.2 Algorithm Design: Multiverse Attention.

Next, we introduce Multiverse Attention to replace the causal attention (Vaswani et al., 2017) in AR-LLMs. Causal attention computes the i-th token’s output with query qi, and keys kj, values vj from positions j ≤ i:

aij = Softmax (qi⊤ ⊙ P(i)) · (kj ⊙ P(j)) + Mij , (1)

0, j ≤ i −∞, otherwise

where Mij =

is the causal mask, and P(i) is the positional embedding for the i-th position.

However, this formulation poses challenges for conceptual parallel generation, as later paths depend on both (i) the key-value (KV) pairs and (ii) the positional indices produced by earlier paths. To address this, we modify both the attention masks and position indices following APE (Yang et al., 2025), as shown in Figure 6b. In Multiverse Attention, each path within the same Process block starts from an identical position and executes independently without accessing others. During the Reduce stage, all paths converge to the same position, which is set to the maximum position reached by any path to avoid negative relative distance, regardless of their variable lengths.

Building on its similarity to causal attention, Multiverse Attention enables (i) Hardware Efficiency:it can preserve training parallelism, and (ii) Data Efficiency: it can be rapidly adapted via fine-tuning on a few samples.

###### 5.3 System Implementation: Multiverse Engine.

To enable truly parallel generation in practical deployments, we introduce Multiverse Engine, an extension of existing inference engines designed for AR models. Specifically, we start from SGLang (Zheng et al., 2023) due to its support for continuous batching and radix attention. These features allow dynamic batch scheduling and flexible KV-cache reuse for Multiverse, two scenarios that frequently occur in the Map and Reduce stages.

The Map stage is automatically triggered when a <Parallel> token is generated. Next, the interpreter counts the number of <Outline> encountered until reaching </Goal>. Based on this count, the engine creates multiple paths executed in parallel, which can be viewed as distinct samples within the same batch. Leveraging radix attention, these paths share the prefix KV cache from the current context. Each path is identified and initiated with “<Path> i” according to its order i in the <Outline> list. After prefilling, all paths are added to the decoding queue for

parallel generation. When a path finishes, either by reaching </Path> or the maximum length, it enters a “zombie” state that releases all resources and waits for the completion of other paths before continuing to the next stage.

The Reduce stage begins once all processing paths have completed their execution. During this stage, the engine merges the KV states from all paths along with the preceding context to form a new sequence. Thanks to the flexible memory layout of the radix cache, indices of KV cache can be seamlessly concatenated without any padding, thereby avoiding both physical memory copying overhead and redundant padding computations. A token <Conclusion>, prefixed with this merged KV cache, is subsequently added to the prefilling queue. Upon completion, the request advances to the decoding queue to continue generation along the newly constructed sequence.

###### 6 Experiments

This section shows the superiority of Multiverse over Autoregression in real-world reasoning tasks. Specifically,

- • In Section 6.2, Multiverse-32B achieves substantial improvements over the Qwen2.5 model by 24.5% after SFT on Multiverse-1K, while matching or exceeding the performance of AR-LLMs on real-world reasoning tasks.
- • In Section 6.3, Multiverse-32B exhibits a superior tradeoff between performance and latency than AR-LLMs. It achieves this by generating more tokens within the same wall-clock time, indicating a more efficient scaling.

###### 6.1 Setup.

Training. We create Multiverse-32B by performing SFT on the Qwen2.5-32B-Instruct model (Qwen, 2024), integrating our Multiverse Attention. The training dataset combines our Multiverse 1K dataset prompted with

“Think step by step and in parallel” and the original sequential data appended by “Think step by step”. We employ a dynamic mixture ratio that progressively shifts from 0:1 (exclusively Autoregressive data) to 1:0 (exclusively Multiverse data) across eight epochs. Fine-tuning took 3 hours on 8 NVIDIA B200 GPUs with PyTorch FSDP.

Evaluation. We measure Multiverse-32B on four reasoning tasks, including AIME24 (Mathematical Association of America, 2024), AIME25 (Mathematical Association of America, 2025), MATH500 (Hendrycks et al., 2021), and GPQA Diamond (Rein et al., 2024). LightEval (Habib et al., 2023) is employed as the evaluation toolkit, powered by our SGLang (Zheng et al., 2023)-based Multiverse Engine. We evaluate our model under two prompting conditions: with and without the phrase “in parallel”, where the latter one is denoted as Multiverse-32B-zero.

Baselines. We compare Multiverse-32B with the Qwen2.5 model and an Autoregressive-32B trained using the same data, but without any control tags or extra Map and Reduce stages. s1-32B and s1.1-32B (Muennighoff et al., 2025) are also included for reference. In addition to pass@1, we measure the degree of parallelism (# parallel) as the ratio between the number of generated tokens and the number of sequentially generated tokens.

###### 6.2 Real-world Reasoning Performance

Table 2 Performance comparison between Multiverse-32B and other 32B AR-LLMs. The pass@1 metric is reported using LightEval (Habib et al., 2023), with results averaging over 8 seeds on AIME. The # parallel computes the ratio between the total number of generated tokens and the actual generation length, measuring the degree of parallelism.

###### AIME24 AIME25 MATH500 GPQA-Diamond Model / Metric pass@1 # parallel pass@1 # parallel pass@1 # parallel pass@1 # parallel

s1-32B 35.4 1.00 25.8 1.00 88.6 1.00 48.0 1.00 s1.1-32B 52.9 1.00 41.7 1.00 93.4 1.00 62.6 1.00

Qwen2.5-32B-Instruct 15.8 1.00 10.4 1.00 80.4 1.00 47.0 1.00 Autoregressive-32B 54.6 1.00 45.0 1.00 92.8 1.00 61.6 1.00

Multiverse-32B-zero 52.1 1.04 44.2 1.05 92.4 1.12 63.6 1.17 Multiverse-32B 53.8 1.18 45.8 1.15 91.8 1.15 60.7 1.17

In Table 2, we report the performance of Multiverse-32B on complex reasoning tasks with 32K contexts, showing improvements of 38%, 35%, 11%, and 14% over the Qwen2.5-32B-Instruct model across the respective benchmarks after fine-tuning. Notably, Multiverse-32B matches or even surpasses the performance of autoregressive models,

as demonstrated by its comparison with Autoregressive-32B. For reference, we also include the results of the s1.132B model trained on the sequential CoT data from which Multiverse-1K is derived. The comparable performance between these models confirms that our data curation pipeline successfully preserves the original data quality.

We also evaluate Multiverse-32B-Zero, a variant prompted without the “think in parallel” instruction. Comparing the two variants reveals distinct performance patterns: Multiverse-32B achieves greater parallelism on AIME tasks, resulting in a slight performance improvement, while Multiverse-32B-Zero performs better on tasks requiring shorter generation sequences, where the model naturally generates in parallel without explicit prompting. This parallelism, measured as the ratio of generated tokens to generation length, aligns with our training strategy, suggesting the potential for controllably switching between AR and Multiverse generation. Notably, the reduced parallelism observed on AIME tasks indicates that the model exhibits less parallelism during longer generation, which we attribute partly to the scarcity of training data exceeding 16K tokens in Multiverse-1K.

###### 6.3 Scaling Performance

To highlight the benefits of parallel generation, we conduct budget control experiments on GPQA-Diamond and MATH500 using the same context length (i.e., approximately the same generation time), varying from 1K to 4K tokens. As illustrated in Figure 7, while longer contexts improved performance for both models, Multiverse-32B generates more tokens within the same context length. This parallel scaling yielded performance improvements of 2.23% on the GPQA-Diamond (with # parallel = 1.17) and 1.51% on the MATH500 (with # parallel = 1.15).

55

| |
|---|

Multiverse

Autoregressive

| |
|---|

50

Performance

| |
|---|

45

40

35

30

1K 1.5K 2K 2.5K 3K 3.5K 4K 4.5K

Generation Length (Time)

(a) GPQA-Diamond

Multiverse

| |
|---|

80

Autoregressive

| |
|---|

Performance

70

| |
|---|

60

| |
|---|

50

40

1K 1.5K 2K 2.5K 3K

Generation Length (Time)

(b) Math500

- Figure 7 Multiverse achieves better performance using a fixed context length, which indicates the same generation time. Here, we report the actual generation length, as some data points terminate before reaching the maximum length.

###### 7 Efficiency Analysis

Having demonstrated Multiverse-32B’s strong scalability and overall performance, we now further analyze the practical efficiency of Multiverse with our engine, showing the potential unlocked through parallel generation.

First, we investigate the relationship between the degree of parallelism and latency per token across various generation lengths (8K, 16K, and 32K), using a batch size of one. The resulting data points, illustrated in Figure 8a, demonstrate that Multiverse enhances generation efficiency by increasing the degree of parallelism. Furthermore, we fit the sampled data points into three inverse curves, one for each. These curves highlight the potential of Multiverse to further reduce latency by encouraging parallelism. Specifically, we identify three key regions based on the sample distribution, demarcated by red lines. The first, encompassing parallelism degrees from 1.0 to 1.3, represents the majority of data points, yielding an average speedup of 18.5%. In the second region, examples show that higher parallelization is achievable, offering acceleration of up to 2.1×. Finally, the third region, characterized by extended lines, demonstrates the promising potential for further improvement with increased parallelism.

Next, we show the speedup achieved by Multiverse-32B with varying degrees of parallelism across different batch sizes, while keeping a fixed 4K output length. The results in Figure 8b indicate that the generation

17.5

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | |y y<br><br>|= 15.43/x + 0.60<br>= 15.43/x + 1.06<br>| |
| | | | | |y 8 1<br><br>|= 15.38/x + 2.14<br><br>K: sampled dat 6K: sampled dat|a a|
| | | | | |3|2K: sampled dat|a|
| | | | | | | | |
| | | | | | | | |

###### Latencypertoken(ms)

15.0

12.5

10.0

7.5

1.0 1.5 2.0 2.5 3.0

# Parallel

(a) Reduced Latency/Token with Increased # Parallel

P=11 P=12 P=13 P=15

1.6

Speedup

1.4

1.2

1.0

2 3 4 5 6 7

Log2(Batch Size)

(b) Stable Speedup Across Varying Batch Size

- Figure 8 Efficiency Analysis. Multiverse can obtain efficiency gains across varying batch sizes based on # Parallel.

process remains memory-bound as the batch size increases from 1 to 128. Therefore, the speedup of Multiverse scales linearly with the degree of parallelism across multiple configurations, showcasing its excellent scalability.

###### 8 Conclusion

This work proposes Multiverse, a natively parallel generative model based on a MapReduce paradigm that internalizes three stages: (i) a Map stage for adaptive task decomposition, (ii) a Process stage for parallel subtask execution, and (iii) a Reduce stage for lossless result synthesis. To build a real-world Multiverse model, we co-design our data, algorithm, and system, enabling a seamless and rapid transfer from AR-LLMs. After fine-tuning on Multiverse-1K, our Multiverse-32B achieves performance comparable to AR-LLMs on real-world reasoning tasks, while achieving better scaling using the same context length due to parallel generation. Moreover, it leads to up to 2× wall-clock speedup per generated token across varying batch sizes, based on the degree of parallelism. We hope that Multiverse can serve as an alternative to Autoregression for generative modeling.

###### 9 Limitations

While Multiverse provides a general framework for generative modeling, its application to diverse data and task types beyond LLM reasoning remains underexplored. Moreover, as Multiverse-32B was trained solely using Supervised Fine-Tuning (SFT), a key direction in future research is to integrate Reinforcement Learning (RL) into training to explore and encourage more parallelism, which in turn would require a more robust Multiverse engine.

###### 10 Broader Impacts

Multiverse significantly boosts GPU utilization by enabling massive parallel generation. This modeling framework is particularly beneficial for small-batch and long-context inference scenarios, leading to substantial reductions in latency and corresponding energy consumption. Furthermore, Multiverse enables economies of scale for difficult but parallelizable tasks, decreasing the time per task unit while maintaining near-constant overall latency, even as task complexity increases. This remarkable scalability showcases its potential to address extremely complex tasks in practice that were previously intractable, offering a promising path towards artificial superintelligence (ASI).

###### Acknowledgement

We thank Zhuoming Chen, Haizhong Zheng, Ranajoy Sadhukhan, Yang Zhou, Songlin Yang, Liliang Ren, Wentao Guo, Ruijie Zhu, Yu Zhang, and Yixin Dong for their constructive feedback on this work, along with the authors of s1 (Muennighoff et al., 2025), SGLang (Zheng et al., 2023), and LightEval (Habib et al., 2023) for their useful codebase. We are also grateful to BitDeer AI Research for providing GPU resources and to Google for supplying Gemini API credits. This research was supported in part by a DGX B200 gift from NVIDIA, a Google Research Award, an Amazon Research Award, Intel, Li Auto, Moffett AI, and the CMU CyLab Seed Fund.

###### References

Alfred V. Aho, John E. Hopcroft, and Jeffrey D. Ullman. The Design and Analysis of Computer Algorithms. AddisonWesley, 1974. ISBN 9780201000290.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Lo¨ıc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-juss`a, David Dale, et al. Large concept models: Language modeling in a sentence representation space. arXiv preprint arXiv:2412.08821, 2024.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher R´e, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Jeffrey Dean and Sanjay Ghemawat. Mapreduce: simplified data processing on large clusters. Communications of the ACM, 51(1):107–113, 2008.

Jeffrey Dean, Sanjay Ghemawat, et al. Mapreduce: simplified data processing on large clusters. In osdi, volume 4, page 5. USA, 2004.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. https://arxiv.org/abs/2501.12948.

Nima Fathi, Torsten Scholak, and Pierre-Andre´ Noe¨l. Unifying autoregressive and diffusion-based sequence generation. arXiv preprint arXiv:2504.06416, 2025.

Guhao Feng, Yihan Geng, Jian Guan, Wei Wu, Liwei Wang, and Di He. Theoretical benefit and limitation of diffusion language model. arXiv preprint arXiv:2502.09622, 2025.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.

Google. Gemini 2.0 flash thinking mode (gemini-2.0-flash-thinking-exp-1219). https://cloud.google.com/vertex-ai/ generative-ai/docs/thinking-mode, 2024. Accessed: 2025-04-22.

Google. Gemini 2.5 (gemini-2.5-pro-preview). https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025a. Accessed: 2025-04-22.

Google. Gemini 2.5: Our most intelligent ai model, March 2025b. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Nathan Habib, Cl´ementine Fourrier, Hynek Kydl´ıˇcek, Thomas Wolf, and Lewis Tunstall. Lighteval: A lightweight framework for llm evaluation, 2023. https://github.com/huggingface/lighteval.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob

Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. David Hounshell. From the American system to mass production, 1800-1932: The development of manufacturing

technology in the United States. Number 4. Jhu Press, 1984. Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Tian Jin, Ellie Y Cheng, Zack Ankner, Nikunj Saunshi, Blake M Elias, Amir Yazdanbakhsh, Jonathan Ragan-Kelley, Suvinay Subramanian, and Michael Carbin. Learning to keep a promise: Scaling language model decoding parallelism with learned asynchronous decoding. arXiv preprint arXiv:2502.11517, 2025.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. In Forty-first International Conference on Machine Learning, 2024.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

- Mathematical Association of America. American Invitational Mathematics Examination 2024, 2024. https:// artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination. Accessed: 202505-14.
- Mathematical Association of America. American Invitational Mathematics Examination 2025, 2025. https:// artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination. Accessed: 202505-14.

John McCarthy. Recursive functions of symbolic expressions and their computation by machine, part i. Communications of the ACM, 3(4):184–195, 1960.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cande`s, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Robert Netting. Smallholders, householders. The ENVIRONMENT in anthropology: A reader in ecology, culture, and sustainable living, 10:14, 1993.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

OpenAI. Introducing openai o3 and o4-mini, April 2025. https://openai.com/index/introducing-o3-and-o4-mini/. OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar,

Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan

Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quin˜onero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. https://arxiv.org/abs/2412.16720.

Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr.

Learning adaptive parallel reasoning with language models. arXiv preprint arXiv:2504.15466, 2025. Qwen. Qwen2: A family of open-source language models by alibaba cloud, 2024. https://github.com/QwenLM/Qwen. Qwen. Qwq-32b: Embracing the power of reinforcement learning, March 2025. https://qwenlm.github.io/blog/

qwq-32b/.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Gleb Rodionov, Roman Garipov, Alina Shutova, George Yakushev, Vage Egiazarian, Anton Sinitsin, Denis Kuznedelev, and Dan Alistarh. Hogwild! inference: Parallel llm generation via concurrent attention. arXiv preprint arXiv:2504.06261, 2025.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. arXiv preprint arXiv:2503.00307, 2025.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022. xAI. Grok 3 beta — the age of reasoning agents, February 2025. https://x.ai/news/grok-3. Xinyu Yang, Tianqi Chen, and Beidi Chen. Ape: Faster and longer context-augmented generation via adaptive parallel

encoding. arXiv preprint arXiv:2502.05431, 2025. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts:

Deliberate problem solving with large language models. Advances in neural information processing systems, 36: 11809–11822, 2023.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025a. https://hkunlp.github.io/blog/2025/dream.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025b.

Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, and Wanli Ouyang. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b. arXiv preprint arXiv:2406.07394, 2024.

Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning. arXiv preprint arXiv:2504.12216, 2025.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Efficiently programming large language models using sglang. 2023.

# Appendix

###### A Prompt of Multiverse Curator

In this section, we release our complete five-stage prompting protocol to create Multiverse-1K, powered by the Gemini 2.5 Pro model. This protocol is engineered to transform any sequential CoT data into Multiverse data.

This protocol starts with a multi-round conversation with the LLM (Stages 1-3) to convert an original reasoning chain into a parallel-structured summary. In Stage 4, both this summary and the original reasoning trajectory are fed to the LLM to repopulate each summarized step with its complete, original details. A content checker then immediately assesses these refilled steps. If the editor distance (e.g., Levenshtein distance between the original trajectory (sori) and its rewritten version (sgen), denoted as d(sori,sgen)) is too high, that step is re-generated. To normalize this, a relative editor distance is calculated to decide if a threshold r is exceeded (set to 0.2 in practice):

d(sori,sgen) max(length(sori),length(sgen))

Relative Editor Distance =

Next, in Stage 5, we transform the output from Stage 4 into a MapReduce-structured reasoning trajectory by inserting the Map and Reduce phases that are generated by Gemini 2.5 Pro. To ensure the structural validity of the data, we perform a grammar check using a customized XML interpreter, which filters out invalid entries and extracts the outermost MapReduce blocks in the remaining valid ones. Finally, each path is rewritten separately to produce fully independent reasoning paths. The prompts used in the entire protocol are as follows:

- STAGE 1: Generating a Summary Tree

###### Main-Step Extraction

Analyze the given reasoning chain (for a math or coding problem) and pull out every major step. Ignore substeps—only list the top-level insights or actions.

Output format

- S1: [First major step]
- S2: [Second major step]
- S3: [Third major step]

... SX: [Description of step X]

... Guidelines

- • Label each top-level step consecutively (‘S1’, ‘S2’, ‘S3’, . . . ).
- • Please capture the entire thought process presented in the reasoning chain, and do not skip any step that includes but not is limited to:

- 1. Initial problem understanding and analysis
- 2. All exploration paths (both successful and unsuccessful)
- 3. Case studies, checks, or tests performed
- 4. Any “aha” or correction (re-evaluation or re-thinking) moments
- 5. The final reasoning that yields the solution

- • Keep each item concise yet descriptive.
- • Do not include any sub-numbering (no ‘S2.1’, etc.).
- • Explicitly split multiple cases or scenarios into different steps. Each case should be allocated an independent step.

###### Substep Extraction

Given the output including all main step from a reasoning chain, break it down into all its internal substeps only if it can be meaningfully subdivided into smaller thought units.

Output format

- S1: [Description of step 1]
- S2: [Description of step 2]

- S2.1 [Description of step 2.1]
- S2.2 [Description of step 2.2]

... S2.10 [Description of step 2.10]

- S3: [Description of step 3]
- S4: [Description of step 4]

... S10: [Description of step 10]

... Guidelines

- • Use the same parent index (‘x’) as the main step (e.g. if breaking down ‘S2’, label ‘S2.1’, ‘S2.2’, . . . ).
- • Capture the entire thought process presented in the reasoning chain, and do not skip any substep that includes but is not limited to:

- 1. Initial problem understanding and analysis
- 2. All exploration paths (both successful and unsuccessful)
- 3. Case studies, checks, or tests performed
- 4. Any “aha” or correction (re-evaluation or re-thinking) moments
- 5. The final reasoning that yields the solution

- • Do not introduce deeper nesting larger than 2 (e.g. ‘S2.1.1’ is not allowed).
- • Explicitly split multiple cases or scenarios into different substeps. Each case should be allocated an independent substep.

- STAGE 2: Identifying Parallel Groups

###### Parallelizing Main Steps

Using only the main steps (S1, S2, . . . ) you extracted in Stage 1, identify all steps or contiguous step groups that can be executed in parallel without violating logical dependencies, and rewrite the plan as a structured parallel execution outline.

- 1. Identify Parallel Groups

- • Find sets of adjacent main steps with no dependencies among them.
- • Label groups P1, P2, . . . and list their step ranges (e.g. [S1+S2, S3], [S4]).

- 2. Rewrite into a Parallel Execution Plan

• Preserve each step’s original wording as much as possible. Output Format: Parallel groups:

- P1: [S1+S2, S3]
- P2: [S4]

... Parallel execution plan:

- P1[parallel reason: ...]: S1+S2: [text of S1 + text of S2]

- S3: [text of S3]

P2[parallel reason: ...]:

- S4: [text of S4]

... Guidelines

- • Coverage: Include every step exactly once, either alone or inside a parallel group.
- • Contiguous Blocks: Combine only adjacent steps into blocks; do not combine non-adjacent steps.
- • Strict Parallelism Only: Build a dependency graph: draw an edge from step A to B if B uses A’s output. A group P_i may include steps (or blocks) only if there are no edges between them. Treat conditional branches as independent tasks.
- • Contiguous Grouping Only: Each parallel group must cover a continuous sequence of steps. Do not parallelize non-adjacent steps.
- • Conciseness: Keep each bullet short and stick closely to the original text.

Parallelizing Substeps

Using only the substeps (S2.1, S2.2, ...) you extracted in Stage 1, identify all substeps or contiguous substep groups can be executed in parallel without violating logical dependencies, and rewrite the plan as a structured parallel execution outline.

- 1. Identify Parallel Groups

- • Find sets of adjacent main steps with no dependencies among them.
- • Label groups P1, P2, . . . and list their step ranges (e.g. [S2.1+S2.2, S2.3], [S3.1]).

- 2. Rewrite into a Parallel Execution Plan

• Preserve each step’s original wording as much as possible. Output Format: Parallel groups: P1: [S2.1+S2.2, S2.3] P2: [S2.4] P2: [S3.1] ... Parallel execution plan:

- P1[parallel reason: ...]: S2.1+S2.2: [text of S2.1 + text of S2.2]

- S2.3: [text of S2.3]

- P2[parallel reason: ...]:

- S3.1: [text of S3.1]

... Guidelines

- • Coverage: Include every substep exactly once, either alone or inside a parallel group.
- • Contiguous Blocks: Combine only adjacent substeps into blocks; do not combine non-adjacent substeps.
- • Strict Parallelism Only: Build an explicit dependency graph in your analysis: draw an edge from substep A to substep B if B uses A’s output or insight. A group Pi may include steps (or contiguous blocks) only if there are no edges between any two steps. In conditional logic, treat the if branch and else branch as independent tasks and parallelize them even though their outputs cannot both occur at runtime.
- • Contiguous Grouping Only: Each parallel group must cover a continuous sequence of steps or blocks. In other words, you may only parallelize adjacent substeps. The occurrence of substeps in parallel groups must follow their original order. For example, P1: [S2.2, S3.1] is not allowed.
- • Conciseness: Keep each bullet short and stick closely to the original text.

- STAGE 3: Reformating into Parallel Structures

###### Get Structured Summary

Please summarize the conversation above by extracting the reasoning steps and substeps in Stage 1 as a tree structure with explicit parallelism annotations following Stage 2.

Output Format

- O1: [Brief summary of top-level step S1] <parallel>[parallel reason: ...]

- O1.1: [Summary of substep S1.1]
- O1.2: [Summary of substep S1.2]

... </parallel> <parallel>[parallel reason: ...]

- O2: [Brief description of top-level step S2] <parallel>[parallel reason: ...]

- O2.1: [Summary of substep S2.1 + Summary of substep S2.2]
- O2.2: [Summary of substep S2.3]

... </parallel>

- O3: [Brief description of top-level step S3] </parallel>
- O4: [Brief description of top-level step S4]

... Guidelines

- • Max depth of nested <parallel> is 2. Do not nest <parallel> tags more deeply than two levels.
- • Max depth of nested numbering is 2. Only use Ox and Ox.y; do not introduce deeper numbering like Ox.y.z.
- • Sequential subpaths stay unexpanded. If a node’s children are purely sequential, list them normally without any <parallel> wrapper.
- • Tag parallel blocks. Wrap only genuinely parallelizable sibling steps in a <parallel>. . . </parallel> block, and include a parallel-reason annotation.
- • Concise summaries. Each step and substep should be described briefly and clearly.
- • Avoid over-splitting. If most children are sequential and only a pair can run in parallel, either leave the group un-split or tag only the truly parallel pair.
- • Group parallelizable sets. You may combine several independent paths into one <parallel> block when they share no dependencies.

- STAGE 4: Refilling Original Details

Refill the Full, Detailed Reasoning Trajectories into the Structured Summary

You will receive an outline that may be incomplete but includes <parallel> tags indicating parallel structures. It contain summaries for several steps and substeps. You will also receive the corresponding original text, where sentences implicitly or explicitly map to hierarchical prefixes (e.g., O1, O1.1, O2) in sequence. Your task is to process the original reasoning chain sequentially to update the outline: replace existing summaries or insert new steps as needed, while preserving the original <parallel> tag structure.

Guidelines:

- • Initialize Structure Start with the structure provided by the input outline, including its text/summaries and all <parallel> tags in their original locations.
- • Read Sentences Sequentially: Process each sentence of the original text one by one, in the exact order they appear.
- • Process Each Sentence:

- 1. Determine the hierarchical prefix associated with this sentence (e.g., O1, O1.1, O2).
- 2. Check if a step or substep with this prefix already exists in the outline.
- 3. If it exists: Replace its current summary with the full original sentence.
- 4. If it does not exist: Insert a new step/substep at the correct hierarchical position (e.g., S1.1 under S1, S2 after S1), using the full original sentence as its content and matching the outline’s indentation.

- • Preserve <parallel> Tags: Keep every existing <parallel> and </parallel> tag exactly where it was in the input outline. Do not add, remove, or relocate any tags.
- • Ensure Correct Output Formatting:

- – Maintain proper hierarchical indentation for all steps and substeps.
- – Each entry must be on its own line, beginning with its prefix (e.g., O1:, O1.1:), followed by the full original sentence.

- • Maintain Completeness: Verify that every sentence from the original reasoning chain has been processed and appears in the updated outline. Do not omit or merge any sentences.

- STAGE 5: Adding MapReduce Structures & Rewriting All Paths

###### Filling Detailed Goal and Conclusion Based on the New Reasoning Trajectory

Based on the generated reasoning chain, your task is to transform it according to the following rules: Output Format [Full reasoning copied from the reasoning chain for the first top-level path] [Full reasoning copied from the reasoning chain for the second top-level path]

... Let’s think in parallel. <Parallel> <Goal>

- Path: [brief, self-contained description of case A]
- Path: [brief, self-contained description of case B]

... </Goal> <Path> [Introductory reasoning for case A] Let’s think in parallel. <Parallel> <Goal>

- Path: [brief, self-contained description of case A.1]
- Path: [brief, self-contained description of case A.2] </Goal> <Path>

- [Full detailed reasoning for case A.1, rewritten clearly and independently]

- </Path> <Path>
- [Full detailed reasoning for case A.2, rewritten clearly and independently] </Path> <Conclusion> [Your concise summary of outcomes from A.1 and A.2] </Conclusion> </Parallel> </Path> <Path> [Full detailed reasoning for case B, rewritten clearly and independently] </Path>

... <Conclusion> [Your concise summary of outcomes from A and B] </Conclusion> </Parallel> [Full detailed reasoning for any remaining paths]

Guidelines

- • Remove all numbering labels (e.g., O1, O2.1) and eliminate any indentation.
- • For each <Parallel>...</Parallel> block:

- – Group every step, substep, and subsubstep belonging to the same parallel branch into a single <Path>...</Path> section.
- – Discard the [parallel reason: ...] annotations.

- • Within each <Parallel> block:

- – Insert <Goal> before the first <Path>, listing each branch as Path: . . . .
- – Insert <Conclusion> after the last <Path>, summarizing each branch’s outcome independently.

- • When multiple <Path> entries stem from the same original sentence or have interdependencies:

- – Rewrite each path separately and completely, ensuring no cross-references.
- – Provide enough context in each <Path> so it stands alone.
- – Fully encapsulate the logical reasoning for each path.

- • Avoid repetition: do not echo the brief descriptions from <Goal> inside the corresponding <Path>, and minimize redundant information across paths.

###### Rewriting Paths in the Structured Reasoning Trajectory

You are given a full structured reasoning trajectory inside a <Parallel> block, consisting of:

- • one <Goal> block with multiple <Outline> elements
- • multiple <Path> blocks
- • one <Conclusion> block. Some <Path> blocks may contain an entire nested <Parallel> structure (from <Parallel> to </Parallel>). These nested blocks should be rewritten using the same rules recursively. For <Goal>:
- • Rewrite each <Outline> into a concise statement of what is being calculated or determined.
- • Remove any content describing how the problem is solved or intermediate reasoning steps. For each <Path>:
- • Keep the original numbering prefix (e.g., ‘1:’, ‘2:’).
- • Rewrite the content as a complete, fluent, and logically self-contained paragraph.
- • Do not use transitional phrases like “First,” “Then,” “Next,” “On the other hand,” etc.
- • If the <Path> contains five or fewer sentences, rewrite them together as a single coherent paragraph, ensuring logical flow and fluency without using transitional phrases.
- • If the <Path> contains more than five sentences: Rewrite the first five sentences together as a single unit, forming a fluent paragraph. For the remaining sentences, rewrite each one individually, based on its meaning, as clear and fluent standalone statements.

- • If the <Path> contains a nested <Parallel> block, apply all these rules recursively to the nested block. Each <Path> must make sense independently, even if it contains a nested reasoning chain. For <Conclusion>:
- • Rewrite the conclusion as the most concise and synthesized summary of the main outcomes from all <Path> blocks.
- • You may combine or compare results from different paths, but keep it succinct and direct. Nested <Parallel>:
- • A nested <Parallel> may appear only as a full block inside a <Path>.
- • If a <Path> contains a nested <Parallel>...</Parallel> block, process that inner block exactly as you would the top-level one:

- – Rewrite the inner <Goal>, <Path>, and <Conclusion> elements accordingly.
- – Maintain the XML structure — do not reindent or alter the tag hierarchy.

Output Format <Parallel> <Goal> <Outline> 1: [concise description of the goal of Path 1] </Outline> <Outline> 2: [concise description of the goal of Path 2] </Outline> </Goal> <Path>

- 1: [self-contained paragraph for Path 1, rewritten sentence by sentence] <Parallel> <Goal> <Outline>

- 1.1: [concise goal for Path 1.1] </Outline> <Outline>
- 1.2: [concise goal for Path 1.2] </Outline> </Goal> <Path>

- 1.1: [rewritten paragraph for Path 1.1] </Path> <Path>
- 1.2: [rewritten paragraph for Path 1.2] </Path> <Conclusion> [summarized results from Paths 1.1 and 1.2] </Conclusion> </Parallel> </Path> <Path>

- 2: [self-contained paragraph for Path 2, rewritten sentence by sentence] </Path> <Conclusion> [summary of outcomes from Path 1 and Path 2] </Conclusion> </Parallel> Guidelines

- • You must preserve all XML tags exactly as they appear — do not remove, rename, or reorder tags.
- • Only modify the inner text of <Outline>, <Path>, and <Conclusion> blocks.
- • Wrap the final output in a single Markdown code block using triple backticks.

###### B Examples in Training Data and Model Generation

We present several examples from our Multiverse-1K dataset, along with the corresponding outputs generated by Multiverse-32B, which was trained on this dataset. Both typical collective and selective branches are included.

###### B.1 Training Examples from Multiverse-1K

- Multiverse Data A: Collective Case

<Parallel> <Goal> <Outline>

- 1: Determine the sum of distances from P1 to P2, P3, and P4. </Outline> <Outline>
- 2: Determine the sum of distances from P2 to P1, P3, and P4. </Outline> <Outline>
- 3: Determine the sum of distances from P3 to P1, P2, and P4. </Outline> <Outline>
- 4: Determine the sum of distances from P4 to P1, P2, and P3. </Outline> </Goal> <Path>

- 1: For point P1(0, 0), the distance to P2(10, 20) is (10 − 0)2 + (20 − 0)2 = √100 + 400 = √500 ≈ 22.36.

- The distance to P3(5, 15) is (5 − 0)2 + (15 − 0)2 = √25 + 225 = √250 ≈ 15.81.

- The distance to P4(12, −6) is (12 − 0)2 + (−6 − 0)2 = √144 + 36 = √180 ≈ 13.42.

The sum is 22.36 + 15.81 + 13.42 ≈ 51.59. </Path> <Path>

- 2: For point P2(10, 20), the distance to P1(0, 0) is (10 − 0)2 + (20 − 0)2 = √500 ≈ 22.36.

- The distance to P3(5, 15) is (10 − 5)2 + (20 − 15)2 = √25 + 25 = √50 ≈ 7.07.

- The distance to P4(12, −6) is (10 − 12)2 + (20 − (−6))2 = √4 + 676 = √680 ≈ 26.08.

The sum is 22.36 + 7.07 + 26.08 ≈ 55.51. </Path> <Path>

- 3: For point P3(5, 15), the distance to P1(0, 0) is (5 − 0)2 + (15 − 0)2 = √250 ≈ 15.81.

The distance to P2(10, 20) is

(5 − 10)2 + (15 − 20)2 = √50 ≈ 7.07.

The distance to P4(12, −6) is

(5 − 12)2 + (15 − (−6))2 = √49 + 441 = √490 ≈ 22.14.

The sum is 15.81 + 7.07 + 22.14 ≈ 45.02. </Path> <Path>

- 4: For point P4(12, −6), the distance to P1(0, 0) is (12 − 0)2 + (−6 − 0)2 = √180 ≈ 13.42.

- The distance to P2(10, 20) is (12 − 10)2 + (−6 − 20)2 = √4 + 676 = √680 ≈ 26.08.

- The distance to P3(5, 15) is (12 − 5)2 + (−6 − 15)2 = √49 + 441 = √490 ≈ 22.14.

The sum is 13.42 + 26.08 + 22.14 ≈ 61.64. </Path> <Conclusion>

The sums of distances from P1, P2, P3, P4 to the other three points are approximately 51.59, 55.51, 45.02, and 61.64, respectively.

</Conclusion> </Parallel>

- Multiverse Data B: Selective Case

<Parallel> <Goal> <Outline>

- 1: Determine if H can be constructed as a countable intersection of open dense sets using Baire Category Theorem (BCT), leveraging the completeness of S2. </Outline> <Outline>
- 2: Determine if H can be constructed step-by-step using Ω’s approximation property to ensure density and K-avoidance for chords, while addressing potential difficulties. </Outline> </Goal> <Path>

- 1: The sphere S2 is a complete metric space, suggesting the possibility of constructing H as a countable intersection of open dense sets; however, the specific method for such a construction is not immediately clear. </Path> <Path>
- 2: A step-by-step construction of H could leverage the given condition to ensure density and that all chords avoid K; H must be dense in S2, meaning it has points in every open subset of S2, and any two points in H must have their connecting chord disjoint from K. The family Ω allows approximation of any two points on S2 with a chord X′Y ′ from Ω. This implies Ω is dense in the space of all chords, with the additional property that its chords avoid K. The closed nature of K means its complement within the unit ball is open, so a chord disjoint from K lies entirely within this open complement. Consequently, Ω comprises chords in the complement of K. The challenge is to construct a dense set H such that all chords connecting points within H avoid K, given that for any X, Y ∈ S2, nearby points X′, Y ′ exist whose chord X′Y ′ is in Ω and thus avoids K. One approach might be to construct H as a countable dense set where each pair’s connecting chord belongs to Ω. Achieving density alone is insufficient; all chords between any two points in H must be in Ω. Directly constructing such an H presents significant difficulties. </Path> <Conclusion>

The Baire Category Theorem approach for constructing H via intersections (Path 1) lacks a clear method. A step-by-step construction (Path 2) highlights the requirements of density and K-avoidance for all chords within H, but faces challenges in maintaining the chord property universally during the construction process.

</Conclusion> </Parallel>

###### B.2 Generated Examples from Multiverse-32B

- Multiverse Generation A: Collective Case

<Parallel> <Goal> <Outline>

- 1: Compute Case 1 count. </Outline> <Outline>
- 2: Compute Case 2 count. </Outline> </Goal> <Path>

- 1: In Case 1 the path pattern is H–V–H–V–H, giving three horizontal and two vertical segments. We distribute 8 units into three positive horizontal segments:

8 − 1 3 − 1

=

7 2

= 21.

We distribute 8 units into two positive vertical segments:

8 − 1 2 − 1

=

7 1

= 7.

- Total for Case 1: 21 × 7 = 147.

</Path> <Path> 2: In Case 2 the pattern is V–H–V–H–V, giving three vertical and two horizontal segments. Distribute 8 units into three positive vertical segments:

7 2

= 21. Distribute 8 units into two positive horizontal segments:

7 1

= 7.

- Total for Case 2: 21 × 7 = 147.

</Path> <Conclusion> Case 1 yields 147 paths; Case 2 yields 147 paths; overall 147 + 147 = 294. </Conclusion> </Parallel>

- Multiverse Generation B: Selective Case

<Parallel> <Goal> <Outline>

- 1: Count horizontal vs. vertical segments. </Outline> <Outline>
- 2: List possible H/V patterns. </Outline> </Goal> <Path>

- 1: A path from (0, 0) to (8, 8) uses exactly eight right moves (R) and eight up moves (U). With five segments alternating direction, starting with R forces the pattern

R − U − R − U − R, giving three horizontal segments and two vertical segments. Conversely, starting with U yields U − R − U − R − U,

for three vertical and two horizontal segments. In either case there are four direction changes (one at each segment boundary), and the segment counts differ by one because there are an odd number of segments. </Path> <Path>

- 2: The total move count is 16 (8R + 8U). If H = 3 and V = 2, then the three horizontal segment lengths h1 + h2 + h3 = 8 with each hi ≥ 1, which has

8 − 1 3 − 1

7 2

=

= 21

compositions. The two vertical segments v1 + v2 = 8 with vj ≥ 1 has

8 − 1 2 − 1

7 1

=

= 7

compositions, for 21 × 7 = 147 ways when starting horizontal. Similarly, if V = 3 and H = 2, there are 21 ways to split 8 into three vertical segments and 7 ways for two horizontal segments, again 147 ways. In total there are 147 + 147 = 294 segment-length configurations consistent with five alternating segments summing to (8, 8).

</Path> <Conclusion> A five-segment path must alternate directions, yielding either (H, V ) = (3, 2) or (2, 3) segments. There are 147 ways to allocate segment lengths in each case, for 294 total valid patterns. </Conclusion> </Parallel>

