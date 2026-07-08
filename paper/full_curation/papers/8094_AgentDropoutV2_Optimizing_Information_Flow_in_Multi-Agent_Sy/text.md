## AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning

# arXiv:2602.23258v2[cs.AI]28May2026

Yutong Wang1* Siyuan Xiong1∗ Xuebo Liu1† Wenkang Zhou1‡ Liang Ding2 Miao Zhang1 Min Zhang1

1Harbin Institute of Technology, Shenzhen 2Alibaba Group {wangyutong,xiongsiyuan}@stu.hit.edu.cn {liuxuebo,zhangmiao,zhangmin2021}@hit.edu.cn zhouwenkang22@mails.ucas.ac.cn liangding.liam@gmail.com

### Abstract

- AgentDropoutV1 (ADv1)
- AgentDropoutV2 (ADv2)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

While Multi-Agent Systems (MAS) excel in complex reasoning, they suffer from the cascading impact of erroneous information from individual agents. Current solutions often resort to rigid structural engineering or expensive finetuning, limiting their adaptability. We propose AgentDropoutV2 (ADv2), a test-time rectifyor-reject pruning framework that dynamically optimizes MAS information flow. Acting as an active firewall, ADv2 intercepts agent outputs and employs a retrieval-augmented rectifier to iteratively correct errors. This rectification is guided by an indicator pool, which is constructed offline by distilling error patterns from historical MAS failure trajectories. Irreparable outputs are subsequently pruned to prevent error propagation. Empirical results demonstrate that ADv2 significantly boosts performance on both fixed and dynamic MAS frameworks, achieving average accuracy gains of 6.39 and 2.28 percentage points on extensive math and code benchmarks, respectively. Furthermore, ADv2 exhibits remarkable adaptivity, dynamically modulating rectification efforts based on task difficulty to resolve a wide spectrum of error patterns. Our code is released at https:

Eliminated

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Intercept

Correct

[Figure 13]

|Output T|
|---|

|Output 1| |
|---|---|
| | |

Rectify

Reject

|Output 0|Rectify|
|---|---|
| | |

···

[Figure 14]

Figure 1: Overview of ADv2 versus ADv1. While ADv1 directly discards potential erroneous agents, ADv2 attempts iterative rectification before elimination.

address more complex scenarios (Li et al., 2023). By harnessing collective intelligence (Zhuge et al., 2024; Tian et al., 2025) and orchestrating cooperative teams (Zhang et al., 2025c; Wang et al., 2026b), MAS achieves remarkable performance in complex tasks such as software development (Hong et al., 2024), ultra-long context handling (Li

- et al., 2024a), and scientific discovery (Ghafarollahi and Buehler, 2025). However, the structural complexity of MAS also renders them susceptible to erroneous outputs from individual participants due to error propagation (Zhang et al., 2025d; Pan et al., 2025b). This necessitates the timely identification and pruning of incorrect information to prevent it from cascading to downstream agents and ultimately compromising the entire task.

To mitigate the impact of errors, current research has predominantly diverged into two main paradigms: Structural Optimization and Parameter Internalization. The former seeks to constrain error pathways by engineering robust communication topologies, such as optimizing directed acyclic graphs (DAG) (Zhang et al., 2025b; Wang et al., 2025b). The latter focuses on enhancing the intrinsic reasoning of agents by fine-tuning them on failure trajectories (Motwani et al., 2025; Zhao

- et al., 2025) or utilizing process-supervision data (Lightman et al., 2024; Zhang et al., 2025a). However, these paradigms share a critical bottleneck: the reliance on offline optimization at the expense

//github.com/TonySY2/AgentDropoutV2.

### 1 Introduction

Large language model (LLM)-based agents have achieved outstanding performance across a wide range of tasks, including reasoning (Yao et al., 2023), planning (Prasad et al., 2024), and action (Park et al., 2023). Despite the sophisticated designs that have enabled these agents to achieve significant gains, the single-model paradigm remains a bottleneck that limits their potential. Consequently, a growing body of research has shifted focus towards designing multi-agent systems (MAS) to

* Equal Contribution. † Xuebo Liu is the corresponding author. ‡ This work was done when Wenkang Zhou was interning

at Harbin Institute of Technology, Shenzhen.

of test-time adaptivity. As illustrated in Figure 1, structural methods like AgentDropoutV1 (ADv1) enforce static connectivity graphs that permanently exclude agents without attempting rectification. Similarly, frozen weights also restrict parameterbased methods from dynamic correction. This static nature prevents salvaging correctable errors during inference, underscoring the urgent need for an active, test-time framework to intercept and resolve failures dynamically.

To this end, we introduce AgentDropoutV2 (ADv2), an MAS information flow optimization framework based on test-time rectify-or-reject pruning. During execution, our method actively intercepts each agent’s output to perform iterative rectification before broadcasting it to downstream successors. Specifically, a dedicated rectifier scrutinizes these outputs using adversarial indicators retrieved from a pool of historical failure patterns, providing targeted feedback for detected errors. If errors persist, the output is pruned to strictly prevent error propagation. Experiments demonstrate that this mechanism significantly enhances MAS performance across diverse mathematical and code generation benchmarks. Extended analyses further confirm the system’s adaptability in dynamically retrieving context-aware indicators and efficiently resolving distinct error patterns. Additionally, the observed correlation between pruning rates and reasoning difficulty positions our framework as a potential task evaluator. Our main contributions are listed as follows:

- • We propose a test-time rectify-or-reject pruning method that intercepts and iteratively corrects agent outputs to effectively block error propagation in MAS, thereby safeguarding performance against cascading degradation.
- • We construct an indicator pool by distilling error patterns from failed MAS trajectories, providing an off-the-shelf knowledge base that encapsulates a broad spectrum of reasoning pitfalls for precise error identification.
- • We demonstrate that our method exhibits robust adaptivity across diverse task complexities and scenarios, confirming its effectiveness and generalization capability as a plug-andplay intervention solution.
- • We design ADv2 as a framework-unaware method that transcends ADv1’s static topology constraints, enabling seamless integration

and dynamic info-flow optimization across both fixed and dynamic MAS environments.

### 2 Methodology

#### 2.1 Preliminary

We formulate the MAS workflow as an ordered sequence of N agents, denoted as S = (A1,A2,...,AN). Each agent is defined as a tuple Ai = (Φi,Ri,Ki), where Φi(·) represents the backbone model serving as the reasoning engine, Ri denotes the static role specification, and Ki represents the dynamic knowledge base containing the history of observable messages (initially Ki = ∅). When an active agent Ai receives an input xi, it utilizes its backbone to generate an output:

oi = Φi(xi,Ri,Ki). (1)

Once oi is generated, it is directly transmitted to the downstream successor agents to update their respective knowledge bases via:

Kj ← Kj ∪ {(Ri,oi)}, (2)

where Aj represents any downstream agent designated to receive the information. This sequential interaction across the workflow dynamically constructs a complete inference trajectory T , which encapsulates the initial task Q, the activated agents, their intermediate responses, and the final outcome, formalized as T = (Q,A1:N,o1:N,Y). The workflow concludes when the execution sequence reaches the terminal output agent AN, which is responsible for aggregating the outputs of all preceding agents. Consequently, the final answer to the user task Q is defined as the output generated by this final agent, namely Y = oN.

We present a test-time rectification framework designed to intercept and refine agent outputs during real-time MAS execution. Specifically, before transmitting the output from an intermediate agent to its downstream successors, we actively intercept the message. A dedicated rectifier then scrutinizes the content for potential errors and attempts to resolve them through an iterative refinement process. If the output remains flawed despite these efforts, it is discarded rather than propagated, ensuring that downstream agents are shielded from unreliable information (as shown in Figure 2).

#### 2.2 Test-Time Rectify-or-Reject Pruning

LLMs inherently struggle with autonomous selfcorrection due to confirmation bias, often blindly

###### Agent 1

Agent 2 Agent 3 Agent 4 Agent N

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Rectified

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Task Answer

···

[Figure 32]

[Figure 33]

Yes, accept & proceed

[Figure 34]

More Iterations

[Figure 35]

[Figure 36]

|Round Output<br><br>[Figure 37]<br><br>[Figure 38]|
|---|

No, regenerate based on feedback

###### ···

···

Round Output

[Figure 39]

Pass rate > ？

[Figure 40]

Agent

[Figure 41]

Max Iteration

[Figure 42]

Reached

[Figure 43]

Judgements Rationales

Feedback

[Figure 44]

[Figure 45]

[Figure 46]

Task Scenarios

[Figure 47]

[Figure 48]

[Figure 49]

- Feedback 1

···

- Feedback 2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Action Types

···

···

Rejected

[Figure 55]

[Figure 56]

[Figure 57]

Rectifier

[Figure 58]

Select Indicators

[Figure 59]

[Figure 60]

Active Indicator Set

[Figure 61]

[Figure 62]

[Figure 63]

|Embedding Model<br><br>[Figure 64]|
|---|

[Figure 65]

[Figure 66]

···

Indicator Pool

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Overlapped with No, update retrieved ones?

[Figure 71]

- §3.1 Test-Time Rectify-or-Reject Mechanism
- §3.2 Failure-Driven Indicator Pool Construction

Query Match top-

[Figure 72]

[Figure 73]

[Figure 74]

Deduplicator

Retrieve

[Figure 75]

[Figure 76]

|New Indicators<br><br>[Figure 77]<br><br>···<br><br>[Figure 78]<br><br>[Figure 79]<br><br>Name Definition Trigger Condition<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>Name Definition Trigger Condition<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|
|---|

Yes, discard

|Erroneous Output<br><br>[Figure 88]| |
|---|---|
| | |

[Figure 89]

|Embedding Model<br><br>[Figure 90]|
|---|

Teacher

···

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Task Answer

###### ···

[Figure 99]

[Figure 100]

[Figure 101]

Agent 1

Agent 2 Agent 3 Agent 4 Agent N

Failed

Figure 2: Overview of the proposed framework. The upper block shows the test-time pipeline for iteratively rectifying agent outputs within the MAS. The lower block demonstrates the offline construction of the indicator pool via failure-driven mining and dual-stage deduplication.

endorsing their own flawed reasoning. To ensure objective refinement, the scrutiny must be grounded in explicit criteria rather than open-ended reflection. We formalize these criteria as adversarial indicators—pre-defined error patterns that serve as targeted inspection rules. Our framework incorporates an Indicator Pool I (whose offline construction is detailed in §2.3), where each indicator is structured as a tuple I = (n,d,c). Specifically, the Name (n) identifies the error type; the Error Definition (d) explicitly delineates the abnormal behavior to be audited (providing the objective standard); and the Trigger Condition (c) specifies the reasoning context where this error typically emerges, enabling precise, situation-aware rule matching.

Two-Stage Indicator Retrieval. To efficiently allocate the most pertinent rules for an agent Ai producing output o(it) at iteration t, we employ a coarse-to-fine retrieval strategy. First, we extract the task scenarios and proposed action types from the reasoning context, encoding them into a query vector qi(t) using an embedding model Memb. We then perform semantic matching against the condition embeddings in I to retrieve the top-Kcand candidates, forming Icand(t) . Subsequently, a dedicated Rectifier Model Φrect analyzes the agent’s input, role, and output to dynamically filter Icand(t) , producing a highly relevant active subset Iact(t) ⊆ Icand(t)

(|Iact(t)| ≤ Kact) for the final auditing.

Rectify-or-Reject Pruning. Based on Iact(t), the rectifier systematically audits the output o(it). For each active indicator Ik = (nk,dk,ck) ∈ Iact(t), Φrect evaluates whether o(it) violates the specific constraint dk, yielding a binary violation flag vk(t) ∈ {0,1} and a diagnostic rationale rk(t):

vk(t),rk(t) = Φrect o(it) | xi,Ri,Ik . (3)

Here, vk(t) = 1 signifies an error detection, subsequently triggering the pruning mechanism to pre-

vent error propagation.

To systematically determine the output’s validity while preventing over-pruning, we define the pass rate p(t) as the proportion of active indicators that

the output successfully satisfies (vk(t) = 0). We introduce a tolerance threshold τpass ∈ [0,1] to govern the gating logic. The rectification trajectory follows a tri-state mechanism derived from p(t):

- • Pass: If the pass rate meets the threshold (p(t) ≥ τpass), the output is accepted directly: oi = o(it).
- • Retry: If the pass rate is insufficient (p(t) <

τpass) and the iteration count t < Tmax, we aggregate the diagnostic rationales from the violated indicators to form a targeted feedback set, which the agent then uses to regenerate its output:

F(t) = {rk(t) | Ik ∈ Iact(t) ∧ vk(t) = 1} (4) o(it+1) = Φi xi,Ri,Ki,F(t) . (5)

• Reject: If errors persist and the pass rate remains sub-threshold at the maximum iteration (p(Tmax) < τpass), the output is discarded (oi = ∅) to prevent error propagation. Ultimately, the final message transmitted to the

successor set N(Ai) is defined as:

o(it) if ∃t ≤ Tmax s.t. p(t) ≥ τpass, ∅ otherwise.

(6)

oi =

Structural Degeneration Fallback. While pruning ensures purity, excessive filtering risks structural degeneration. If the remaining message count falls below a safety threshold γ, the MAS is deemed to have lost its reasoning integrity. To prevent fragmented reasoning from a sparse context, we trigger a system-wide reset to re-execute the MAS without the rectify-or-reject mechanism.

The pseudo-code for the test-time rectify-orreject pruning is provided in Appendix A.1, and detailed prompt specifications are listed in Appendix A.2. Additionally, a comprehensive case study demonstrating the rectification process is presented in Appendix A.3.

2.3 Failure-Driven Indicator Pool Construction

Just as mature organizations rely on institutional memory, which codifies lessons learned from past projects to prevent the recurrence of known pitfalls, our framework necessitates a structured repository of error patterns. Blindly correcting errors without understanding their origins is inefficient; effective rectification requires a reference to historical mistakes. To this end, we construct a repository of adversarial indicators by mining historical failure cases. This process transforms raw failure trajectories into a structured knowledge base, serving as a comprehensive handbook of prohibitions to guide the agent’s real-time rectification.

Offline Indicator Mining As illustrated in the lower block of Figure 2, we focus on collecting execution strategies where the MAS fails to deliver the correct solution. Let Dsrc = {Q,Y∗} denote the source dataset, where Q and Y∗ represent the input query and the corresponding ground-truth answer, respectively. For each instance, we conduct a full inference roll-out to obtain the MAS execution trajectory T = (Q,A1:N,o1:N,Y). We collect failure cases where the solution Y diverges from the ground truth Y∗ into a failure set Dfail. A teacher model Φteach then scrutinizes individual

agents within Dfail. Upon detecting a deviation in agent Ai’s output oi given its role Ri and the overall task Q, Φteach synthesizes a set of indicators:

Inew = Φteach (T ,Y∗,Ri,oi). (7)

Redundancy Elimination. To prevent duplicate constraints from dominating the top-Kact retrieval and to maintain a compact, high-entropy pool I, we employ a two-stage deduplication process for newly generated indicators Inew. First, we encode the concatenation of its description and condition into a semantic vector vnew = Memb(dnew ⊕ cnew) to retrieve the Kdedup most similar existing indicators Isim ⊂ I via cosine similarity. Subsequently, a deduplication LLM Φdedup evaluates Inew against Isim, appending it to I only if it represents a strictly novel error pattern.

Examples of the constructed indicators are provided in Appendix A.2.

### 3 Experiment

#### 3.1 Experimental Setup

MAS Framework. To comprehensively evaluate our method, we implement it across two distinct MAS infrastructures: (1) Fixed-MAS: Following Wang et al. (2025b), agents are organized within a Directed Acyclic Graph (DAG), executing workflows statically according to a topological sort. (2) Dynamic-MAS: Utilizing AutoGen’s (Wu et al., 2024) SelectorGroupChat1 framework, where a selector iteratively routes tasks within a globally broadcasted context to reach a final decision.

Baselines. We evaluate our method against several representative paradigms of test-time verification: (1) Self-Refine Style (Madaan et al., 2023): Prompting the reasoner agent to independently review and revise its output at each reasoning round. (2) PRM-guided Search: Utilizing Qwen2.5-Math-PRM-7B (Zhang et al., 2025f) to score three candidate outputs per step, retaining only the highest-reward trajectory. (3) MultiTAG Strategy (Yao and Yadav, 2025): An earlystopping majority vote mechanism that halts execution once the leading answer surpasses the runnerup by more than two votes. Additionally, we include ADv1 as a baseline within the fixed framework, given its inherent restriction to static, graphbased MAS topologies.

1https://microsoft.github.io/autogen/stable/userguide/agentchat-user-guide/selector-group-chat.html

Easy Tasks Hard Tasks

Method

All Avg GSM8K MATH500 AQuA AMC23 Avg OlymB AIME24 AIME25 OlymE OlymH Avg

Single Agent 87.64 74.80 83.86 62.50 77.20 47.56 13.33 20.00 20.00 16.00 23.38 47.30 + CoT 93.71 79.40 84.65 67.50 81.32 43.85 20.00 23.33 24.00 15.00 25.24 50.16

Fixed-MAS 93.18 77.40 85.04 65.00 80.16 49.62 26.67 20.00 31.25 17.50 29.01 51.74 + Self-Refine 93.03 77.00 83.46 65.00 79.62 51.30 23.33 20.00 26.25 26.25 29.43 51.74 + PRM 94.84 80.80 84.25 70.00 82.47 49.31 26.67 20.00 21.25 16.25 26.70 51.49 + Multi-TAG 93.78 76.20 83.07 70.00 80.76 48.55 26.67 23.33 20.00 17.50 27.21 51.01 + ADv1 93.56 78.20 83.86 75.00 82.66 49.16 33.33 13.33 25.00 20.00 28.16 52.38 + ADv2 93.63 77.00 84.65 67.50 80.70 49.16 30.00 23.33 32.50 23.75 31.75 53.50

Dynamic-MAS 91.66 78.00 85.43 62.50 79.40 48.15 30.00 20.00 26.00 16.00 28.03 50.86 + Self-Refine 91.51 83.40 85.04 62.50 80.61 49.19 23.33 10.00 20.00 15.00 23.50 48.89 + PRM 93.33 83.80 85.43 67.50 82.52 51.70 30.00 16.67 24.00 15.00 27.47 51.94 + Multi-TAG 92.04 81.00 85.83 62.50 80.34 48.74 30.00 16.67 27.00 18.00 28.08 51.31 + ADv2 91.66 79.60 83.86 70.00 81.28 52.44 30.00 26.67 32.00 17.00 31.62 53.69

- Table 1: Performance comparison of our method against baseline reasoning techniques across mathematical domain benchmarks, conducted within the fixed and dynamic MAS frameworks. The blue section (left) represents relatively easy tasks, while the orange section (right) indicates harder ones. Notably, ADv2 demonstrates the most significant performance improvements on these harder datasets. “OlymB”, “OlymE”, and “OlymH” represent OlympiadBench, OlymMATH Easy, and OlymMATH Hard, respectively.

Backbone Models. We adopt Qwen3.5-9B (Qwen Team, 2026)2 as the backbone of the Dynamic-MAS selector. For the reasoning LLM encompassing all participants and rectifiers, we deploy Qwen3-8B (Team, 2025) and Qwen3-4B, configured with the thinking mode explicitly disabled. For the offline indicator pool construction process, GPT-5.4-2026-03-053 and GPT-4.1-mini2025-0414 serve as the base for the teacher and deduplicator, respectively. Qwen3-Embedding-8B is adopted as the embedding model Memb.

Datasets. For mathematical reasoning, we employ nine benchmarks spanning a spectrum of difficulty levels, including GSM8K (Cobbe et al., 2021), MATH-500 (Lightman et al., 2024), AQuA (Patel et al., 2021), AMC234, OlympiadBench (He et al., 2024), OlymMATH Easy, OlymMATH Hard (Sun et al., 2025), AIME24 (Zhang and Math-AI, 2024), and AIME25 (Zhang and Math-AI, 2025). For code generation capabilities, we assess the model on four established datasets: MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), CodeContests (Li et al., 2022), and LiveCodeBenchV1 (Jain et al., 2025). Regarding the indicator pool construction, we sample trajectories to distill domainspecific adversarial indicators using the training splits of MATH and AQuA for the mathematical domain, and the training sets of MBPP, KodCode (Xu et al., 2025), and CodeContests for the code domain. Detailed statistics for each dataset and the

- 2https://huggingface.co/Qwen/Qwen3.5-9B
- 3https://developers.openai.com/api/docs/models/gpt-5.4
- 4https://huggingface.co/datasets/math-ai/amc23

indicator pool are provided in Appendix A.4.

Hyper-Parameters. We set the max chat turns of SelectorGroupChat to 6, and the max reflection turns Tmax to 3. We set Kcand to 20 and Kact to 5 during indicator retrieval, and set Kdedup to 20 during the deduplication process of pool construction. The pass rate tolerance threshold τpass is set to 60%. The temperature of the rectifier is set to 0, and the others remain 0.7.

#### 3.2 Main Results

Enhancements in Mathematical Reasoning. Table 1 details performance across mathematical benchmarks. Integrating ADv2 successfully elevates both the vanilla Fixed-MAS and DynamicMAS frameworks, achieving the highest overall average accuracies (53.50% and 53.69%). ADv2 particularly excels on hard, Olympiad-level datasets. Unlike Multi-TAG’s passive consensus or PRM’s rigid step-wise scoring, ADv2 actively intercepts and rectifies complex logical flaws, driving substantial gains on intricate tasks like OlymMATH Hard and AIME25, even though its indicator pool is distilled entirely from simpler foundational datasets. Crucially, ADv2 overcomes the architectural limitations of ADv1. While ADv1’s structural pruning strictly restricts it to static topologies, ADv2 is a framework-unaware, plug-and-play module. This enables its seamless integration into DynamicMAS environments, consistently optimizing information flow regardless of system structure.

Indicator Portability Across Models. We directly deploy the indicator pool mined from a

###### Method GSM8K MATH500 AQuA AMC23 OlymB AIME24 AIME25 OlymE OlymH Average

Single Agent 87.64 79.40 83.86 62.50 52.15 23.33 20.00 20.00 16.00 49.43 Dynamic-MAS 93.48 80.80 86.22 72.50 51.41 26.67 20.00 31.00 11.00 52.56

###### + ADv2 93.93 82.20 87.01 77.50 53.93 43.33 26.67 27.00 15.00 56.29

- Table 2: Performance comparison using Qwen3-14B as the backbone. To demonstrate cross-model transferability, we directly apply the indicator pool generated by a Qwen3-8B-based MAS to the Qwen3-14B system.

Method MBPP HumE CodeC LiveC Average

Single Agent 61.87 85.71 7.27 29.50 46.09 Dynamic-MAS 65.76 85.09 6.67 29.00 46.63

+ Self-Refine 64.20 83.23 6.06 32.50 46.50 + PRM 65.76 81.37 5.45 33.00 46.39 + Multi-TAG 66.15 80.75 6.06 29.50 45.61 + ADv2 68.48 85.71 7.27 32.00 48.37

- Table 3: Performance comparison of our method against baselines across code domain benchmarks. “HumE”, “CodeC”, “LiveC” represent HumanEval, CodeContests, and LiveCodeBench, respectively.

fail to fully rectify reasoning flaws, excessive retries may induce over-correction or reasoning drift. Consequently, Tmax = 3 strikes the optimal balance between error mitigation thoroughness and system stability.

Impact of Retrieved Indicator Count. Next, we explored the impact of the number of retrieved indicators as shown in Block II of Table 4. Both reducing the retrieved count to Kact = 3 and increasing it to Kact = 7 degrade performance compared to the optimal setting of Kact = 5. This indicates that while agents benefit from diverse failure patterns, providing an excessive number of indicators results in information overload, distracting the model with less relevant constraints rather than aiding the reasoning process.

Qwen3-8B-based MAS onto a Qwen3-14B system. As shown in Table 2, ADv2 yields robust performance gains, elevating the average accuracy from 52.56% to 56.29%. This confirms the high transferability of the extracted reasoning pitfalls, thereby validating a cost-efficient paradigm where lightweight models construct reusable knowledge bases to supervise more capable target models without redundant mining.

Impact of Pass Rate Threshold. For the pass rate threshold τpass (Block III), the default 60% achieves the highest accuracy. Lowering it to 40% permits error propagation, whereas a strict 100% requirement (zero-tolerance strategy) causes overrejection. Thus, 60% optimally balances quality control with generative flexibility.

#### Enhancements in Code Generation. Table

- 3 presents the evaluation across code generation benchmarks. While the vanilla MAS framework yields marginal average improvements over the Single Agent baseline (46.63% vs. 46.09%), integrating ADv2 consistently elevates the overall performance to 48.37%. Notably, ADv2 drives substantial gains on practical and complex datasets, surging to 68.48% on MBPP and 32.00% on LiveCodeBench. This demonstrates that our rectifyor-reject mechanism and indicator pool are highly effective in the programming domain, successfully intercepting and mitigating complex logical and syntactical errors in the workflows.
- 4 Analysis

Necessity of Pool Deduplication. We examine the necessity of dual-stage deduplication in Block IV of Table 4. Omitting this process drops the average accuracy to 52.32%, indicating that redundant indicator variations overcrowd the retrieved top-k slots. This lack of diversity blinds the agent to other distinct error patterns, confirming the importance of our compact, high-entropy pool construction.

Impact of Indicator Retrieval Mechanism To validate that performance gains stem from relevant guidance, we replace the retrieved indicators with 1 to 5 randomly sampled constraints from the pool. As shown in Block V of Table 4, this introduces confounding noise and substantially degrades the average accuracy to 51.65%. This demonstrates that strict semantic relevance is essential for accurately locating specific error patterns. Building on this need for relevant guidance, our framework also accommodates zero-shot scenarios where a domain-specific pool is unavailable. By utilizing a single universally applicable indicator that always

#### 4.1 Ablation Study

Impact of Rectification Iteration Rounds. We first examine the impact of the rectification iteration budget Tmax on overall performance. As reported in Block I of Table 4, either decreasing it to 2 or increasing it to 4 results in performance degradation. This indicates that while insufficient iterations

Method GSM8K MATH500 AQuA AMC23 OlymB OlymE OlymH AIME24 AIME25 Average ADv2 91.66 79.60 83.86 70.00 52.44 32.00 17.00 30.00 26.67 53.69

- (I) Rectification Iteration Rounds (Tmax, Default: 3)

- 2 Iterations 91.74 80.80 86.22 62.50 49.63 25.00 18.00 33.33 23.33 52.28 4 Iterations 90.83 79.60 82.28 75.00 48.00 29.00 11.00 30.00 20.00 51.75

(II) Number of Retrieved Indicators (Kact, Default: 5)

- 3 Indicators 90.83 81.80 84.65 62.50 50.52 30.00 16.00 30.00 26.67 52.55 7 Indicators 91.13 77.20 82.68 72.50 50.96 24.00 14.00 23.33 26.67 51.39

- (III) Pass Rate Tolerance Threshold (τpass, Default: 60%)

40% 91.36 80.60 83.07 70.00 49.78 26.00 14.00 26.67 20.00 51.27 100% 91.66 79.00 83.86 70.00 50.96 25.00 17.00 30.00 23.33 52.31

- (IV) Indicator Pool Deduplication

w/o Deduplication 91.36 80.60 86.22 75.00 50.37 31.00 13.00 26.67 16.67 52.32

- (V) Indicator Retrieval Mechanism

Random 1-5 91.05 81.20 85.04 70.00 51.85 24.00 15.00 26.67 20.00 51.65 w/o Indicator Pool 89.92 81.20 83.86 70.00 50.81 26.00 16.00 36.67 16.67 52.35

Table 4: Results of the ablation study.

conducts overall logical correctness checks (see Appendix A.2 for more details), our method still remains functional and beneficial even without prior failure pattern mining.

#### 4.2 Iteration Dynamics and Adaptability

To evaluate adaptability, we analyze iteration round distributions across varying difficulties (Figure 3). “Pass @ k-th” indicates outputs accepted at iteration k, while “Rejected” denotes failures after maximum retries. Task complexity strongly correlates with rectification depth: simpler datasets (e.g., GSM8K) exhibit 94.2% immediate acceptance, whereas complex tasks like OlymMATH Hard shift significantly toward multi-round rectifications, with rejections exceeding 6.3%. This demonstrates that ADv2 dynamically modulates intervention intensity—conserving resources on simple queries while sustaining effort for intricate errors. Moreover, this strong correlation allows our framework to double as a potential difficulty evaluator, where the rectification depth and rejection rate serve as quantifiable proxies for task complexity.

#### 4.3 Distribution of Retrieved Indicators

To assess indicator pool utility, we analyze the pairwise Jaccard similarity of the top-10 retrieved indicators across benchmarks (Figure 4). The heatmap reveals distinct block-wise correlations, indicating that tasks with similar reasoning demands share common failure modes. For instance, foundational datasets like GSM8K and AQuA share a 0.43 similarity, while harder benchmarks such as OlymMATH Easy and AIME24 also exhibit a high overlap. Conversely, overlap drops precipitously be-

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 3: Distribution of rectification iterations across different benchmarks.

tween disparate difficulty levels, such as GSM8K and OlymMATH Hard (0.11). This confirms that error patterns are highly task-dependent, validating both our pool’s diverse coverage and the retrieval mechanism’s precision in isolating context-aware constraints for unique domains.

4.4 Trade-off Between Performance and Cost We evaluate the token-accuracy trade-off on math benchmarks as shown in Figure 5. Inherently, as a test-time scaling method, our ADv2 framework consumes more tokens than the vanilla AutoGen framework, explicitly allocated to our rectify-orreject mechanism. By dynamically intercepting faulty reasoning and preventing error propagation across agents, the increased token expenditure directly yields a substantial accuracy surge from 50.86% to 53.69%. Notably, while other two testtime scaling methods (Self-Refine and PRM) consume comparable token budgets, their overall performance remains significantly inferior to our approach. This confirms that exchanging higher token consumption for rigorous error mitigation is a highly justified and effective trade-off in complex scenarios.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 102]

Figure 4: Jaccard similarity between the set of ten most frequently used indicators across different benchmarks.

### 5 Related Work

As MAS scales to handle complex tasks, they become increasingly vulnerable to error propagation, where individual mistakes amplify downstream and disrupt the entire workflow. To address this, prior research has primarily focused on three strategies: (1) robust architecture design, (2) error monitoring, and (3) utilization of inference trajectories.

Robust MAS Architectures. To mitigate error propagation, existing research attempts to engineer robust MAS structures. Several studies explicitly model MAS as optimizable graphs, employing learning or search algorithms to identify superior topologies (Zhuge et al., 2024; Zhang et al., 2025c; Wang et al., 2025b). Similarly, sparse communication effectively reduces noise disturbance (Li et al., 2024b). Furthermore, advanced orchestration and routing strategies that construct specialized cooperative teams can suppress errors originating from underperforming agents (Dang et al., 2025; Zhang et al., 2025e; Wang et al., 2026a; Ong et al., 2025).

Error Monitoring Mechanisms. To prevent error cascading, these methods deploy monitors to detect workflow anomalies. Graph-based approaches treat information flow and topology as signals, utilizing anomaly detectors to capture abnormal patterns (Wang et al., 2025a; Zhou et al., 2025; Pan et al., 2025a). Meanwhile, test-time rectification offers efficient intervention via an “interceptdetect-correct” process for system messages (Xiang et al., 2024; Chen et al., 2025b; Luo et al., 2025). Conversely, error attribution methods conduct root cause analysis to identify specific agents responsible for introducing hallucinations upon task failure (Zhang et al., 2025d; Pan et al., 2025b; Zhang et al., 2025a; Ge et al., 2025).

Utilization of Inference Trajectories. These approaches enhance MAS reliability by leverag-

x

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Figure 5: Average Accuracy versus token consumption on Math benchmarks by the dynamic MAS framework.

ing execution trajectories to construct preference or contrastive data for training key components (Chen et al., 2025a; Motwani et al., 2025; Zhao et al., 2025). Process-aware variants verify intermediate steps to provide fine-grained supervision, preventing models from adopting locally plausible but globally incorrect paths (Zelikman et al., 2022; Lightman et al., 2024). Additionally, mining failure or exploration trajectories as hard negatives strengthens preference optimization, rendering the system robust against misleading intermediate states (Song et al., 2024; Aksitov et al., 2024).

Our framework integrates these paradigms to overcome their limitations. Unlike rigid structural designs, our approach serves as a model-agnostic, plug-and-play module adaptable to diverse frameworks. We advance error monitoring from passive detection to active rectification, ensuring real-time stability via feedback-driven reflection. Finally, leveraging trajectory utilization, we distill historical failures into an adversarial indicator pool, providing precise, prior-guided online supervision.

### Conclusion

In this paper, we introduced AgentDropoutV2, a novel framework designed to optimize information flow in MAS via test-time rectify-or-reject pruning. By mining historical failure trajectories, we constructed an indicator pool that encapsulates domainspecific error patterns. During test-time inference, our framework actively intercepts agent outputs, retrieves pertinent indicators, and enforces an iterative refinement process to resolve latent errors before they propagate. Experimental results demonstrate that this mechanism effectively cleanses the information flow, thereby significantly enhancing system accuracy. Furthermore, our analysis confirms that the indicator retrieval and rectification processes exhibit strong adaptivity to varying task difficulties, along with robust transferability across different domains and backbone models.

### Limitations

While ADv2 demonstrates significant improvements in MAS reasoning, we acknowledge two primary limitations. First, ADv2 incurs higher inference token consumption due to its iterative rectify-or-reject mechanism. However, this aligns with the test-time scaling paradigm which trades inference compute for enhanced accuracy. Besides, our plug-and-play indicator pools effectively eliminate the massive implicit computational costs associated with model fine-tuning. Second, ADv2 occasionally trails lightweight baselines (e.g., CoT or Self-Refine) on simpler tasks like GSM8K. On such foundational datasets, the backbone model’s high intrinsic capability renders straightforward reasoning strategies highly efficient, whereas our aggressive interventions may introduce slight reasoning overhead. Nevertheless, as task complexity scales beyond the model’s intrinsic limits, these simple methods falter, which is precisely where our robust framework excels. Importantly, even on these simpler tasks, integrating ADv2 consistently elevates performance compared to the unaugmented vanilla MAS frameworks.

### References

Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, Manzil Zaheer, Felix Yu, and Sanjiv Kumar. 2024. ReST meets react: Self-improvement for multistep reasoning LLM agent. In ICLR 2024 Workshop on Large Language Model (LLM) Agents.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1 others. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Weize Chen, Jiarui Yuan, Chen Qian, Cheng Yang, Zhiyuan Liu, and Maosong Sun. 2025a. Optima: Optimizing effectiveness and efficiency for LLMbased multi-agent system. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11534–11557, Vienna, Austria. Association for Computational Linguistics.

Zhaorun Chen, Mintong Kang, and Bo Li. 2025b. Shieldagent: Shielding agents via verifiable safety

policy reasoning. In Forty-second International Conference on Machine Learning.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, and Maosong Sun. 2025. Multi-agent collaboration via evolving orchestration. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yu Ge, Linna Xie, Zhong Li, Yu Pei, and Tian Zhang. 2025. Who is introducing the failure? automatically attributing failures of multi-agent systems via spectrum analysis. arXiv preprint arXiv:2509.13782.

Alireza Ghafarollahi and Markus J Buehler. 2025. Sciagents: automating scientific discovery through bioinspired multi-agent intelligent graph reasoning. Advanced Materials, 37(22):2413523.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, and 1 others. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand. Association for Computational Linguistics.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, and 1 others. 2024. Metagpt: Meta programming for A multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. CAMEL: communicative agents for "mind" exploration of large language model society. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, and 1 others. 2024a. GraphReader: Building graph-based agent to enhance long-context abilities of large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12758–12786, Miami, Florida, USA. Association for Computational Linguistics.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, and 1 others. 2022. Competition-level code generation with alphacode. Science, 378(6624):1092–1097.

Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, and Eugene Ie. 2024b. Improving multi-agent debate with sparse communication topology. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7281–7294, Miami, Florida, USA. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, and Chaowei Xiao. 2025. AGrail: A lifelong agent guardrail with effective and adaptive safety detection. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8104–8139, Vienna, Austria. Association for Computational Linguistics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594.

Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Ivan Laptev, Philip Torr, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. 2025. MALT: Improving reasoning with multi-agent LLM training. In Workshop on Reasoning and Planning for Large Language Models.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations.

Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, and Shirui Pan. 2025a. Explainable and fine-grained

safeguarding of llm multi-agent systems via bilevel graph anomaly detection. arXiv preprint arXiv:2512.18733.

Melissa Z Pan, Mert Cemri, Lakshya A Agrawal, Shuyi Yang, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Kannan Ramchandran, Dan Klein, Joseph E. Gonzalez, Matei Zaharia, and Ion Stoica. 2025b. Why do multiagent systems fail? In ICLR 2025 Workshop on Building Trust in Language Models and Applications.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, Online. Association for Computational Linguistics.

Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. 2024. ADaPT: As-needed decomposition and planning with language models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 4226–4252, Mexico City, Mexico. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. 2024. Trial and error: Exploration-based trajectory optimization of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7584–7600, Bangkok, Thailand. Association for Computational Linguistics.

Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, Lei Fang, and Ji-Rong Wen. 2025. Challenging the boundaries of reasoning: An olympiad-level math benchmark for large language models. arXiv preprint arXiv:2503.21380.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chunhao Tian, Yutong Wang, Xuebo Liu, Zhexuan Wang, Liang Ding, Miao Zhang, and Min Zhang. 2025. AgentInit: Initializing LLM-based multi-agent systems via diversity and expertise orchestration for effective and efficient collaboration. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 11870–11902, Suzhou, China. Association for Computational Linguistics.

Jingbo Wang, Sendong Zhao, Jiatong Liu, Haochun Wang, Wanting Li, Bing Qin, and Ting Liu. 2026a. Orchestrating intelligence: Confidence-aware routing for efficient multi-agent collaboration across multiscale models. arXiv preprint arXiv:2601.04861.

Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, and Yang Wang. 2025a. G-safeguard: A topology-guided security lens and treatment on LLM-based multiagent systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7261–7276, Vienna, Austria. Association for Computational Linguistics.

Zhexuan Wang, Xuebo Liu, Li Wang, Zifei Shan, Yutong Wang, Zhenxi Song, and Min Zhang. 2026b. Maspo: Joint prompt optimization for llm-based multi-agent systems. Preprint, arXiv:2605.06623.

Zhexuan Wang, Yutong Wang, Xuebo Liu, Liang Ding, Miao Zhang, Jie Liu, and Min Zhang. 2025b. AgentDropout: Dynamic agent elimination for tokenefficient and high-performance LLM-based multiagent collaboration. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24013– 24035, Vienna, Austria. Association for Computational Linguistics.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, and 1 others. 2024. Autogen: Enabling next-gen llm applications via multiagent conversations. In First Conference on Language Modeling.

Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Carl Yang, and 1 others. 2024. Guardagent: Safeguard llm agents by a guard agent via knowledge-enabled reasoning. arXiv preprint arXiv:2406.09187.

Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. 2025. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding.

Bohan Yao and Vikas Yadav. 2025. Diverse multi-tool aggregation with large language models for enhanced math reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 25264–25282, Suzhou, China. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. Star: Bootstrapping reasoning with

reasoning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. 2025a. Agentracer: Who is inducing failure in the llm agentic systems? arXiv preprint arXiv:2509.03312.

Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. 2025b. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, and Dawei Cheng. 2025c. G-designer: Architecting multi-agent communication topologies via graph neural networks. In ICLR 2025 Workshop on Foundation Models in the Wild.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025d. Which agent causes task failures and when? on automated failure attribution of LLM multiagent systems. In Forty-second International Conference on Machine Learning.

Wentao Zhang, Ce Cui, Yilei Zhao, Yang Liu, and Bo An. 2025e. Agentorchestra: A hierarchical multiagent framework for general-purpose task solving. arXiv preprint arXiv:2506.12508.

- Yifan Zhang and Team Math-AI. 2024. American invi-

- tational mathematics examination (aime) 2024.

Yifan Zhang and Team Math-AI. 2025. American invi-

- tational mathematics examination (aime) 2025.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025f. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301.

Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, and James Zou. 2025. Sirius: Self-improving multi-agent systems via bootstrapped reasoning. In Workshop on Reasoning and Planning for Large Language Models.

Jialong Zhou, Lichao Wang, and Xiao Yang. 2025. GUARDIAN: Safeguarding LLM multi-agent collaborations with temporal graph modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. 2024. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

### A Appendix

#### A.1 Pseudo Codes

Algorithm 1 outlines the pseudo-code for our rectify-or-reject pruning. During MAS execution, the output of each agent is actively intercepted to undergo the rectification process. First, a candidate indicator set is retrieved based on semantic similarity to serve as a reference for potential error patterns, and then the rectifier model select the final active indicators from this set (Lines 6-9). A rectifier model then scrutinizes the output against each retrieved indicator, generating diagnostic rationales as feedback whenever a specific constraint is violated (Lines 11-15). Subsequently, the algorithm employs a tri-state gating mechanism based on the evaluation results, terminating the iteration if the output passes all checks or if the iteration budget is exhausted (Lines 16-26). Upon successful verification, the qualified output is propagated to successor agents (Lines 27-29). Finally, if the resulting information flow becomes critically sparse, a global fallback process is triggered to reset the system and re-initialize execution from scratch (Lines 31-33).

The pseudo-code for the Failure-Driven Indicator Pool Construction is outlined in Algorithm 2. The process begins by iterating through the source dataset to collect execution trajectories where the MAS fails to deliver the correct solution (Lines 34). Subsequently, a teacher model scrutinizes these failure instances, synthesizing candidate indicators that capture the specific error patterns exhibited by individual agents (Lines 5-6). To prevent repository bloating, a redundancy elimination mechanism is applied to each candidate. The algorithm first encodes the new indicator into a semantic vector to retrieve the most similar existing constraints (Lines 7-9). A deduplication model then verifies the novelty of the candidate, admitting it into the global pool only if it represents a distinct and previously unrecorded error type (Lines 10-12).

#### A.2 Indicator & Prompt Design

Indicator Design. Figure 6 displays an example from our constructed indicator pool. This specific indicator is tailored to verify the precision of square

root calculations (a detailed application case is provided in Appendix A.3). For scenarios where a pre-defined indicator pool is unavailable, we design a general-purpose indicator, as illustrated in Figure 7.

Prompt Design. The prompt templates for the rectifier of math and code domains are presented in Figure 9 and Figure 10. Additionally, Figure 11 depicts the prompt template for the teacher model, which is responsible for generating new indicators based on failed MAS execution trajectories.

#### A.3 Case Study

This case study exemplifies the framework’s capability to navigate complex constraint satisfaction problems through a rectify-or-reject dialectical process. The agent was tasked with the math problem shown in Figure 12. The initial solver output (Figure 13) and the first retry (Figure 15) both end at the same wrong answer, 18 + 7π, because both use the internal angle of the regular nonagon for the offset-boundary arcs. The first audit (Figure 14) rejects the response but only moves the solver away from a quarter-circle assumption; the second audit (Figure 16) then isolates the decisive angle-type error and directs the solver to the external angle 2π/9. After applying that feedback, the solver changes the arc contribution from 7π to 2π, produces

|18 + 2π|
|---|

, and passes all five selected indicators. This trajectory demonstrates the robustness of the rectify-or-reject mechanism in stabilizing reasoning through iterative, multi-dimensional constraint enforcement.

#### A.4 Dataset Statistics

Table 5 lists the detailed statistics of the size of the datasets and the constructed indicator pool. The indicator pool for the math domain is constructed on the failed MAS trajectories on the sampled instances from the MATH and AQuA training sets, while the pool for the code domain is derived from similar failed trajectories on the MBPP, KodCode, and CodeContests training sets.

Algorithm 1: Test-Time rectify-or-reject Pruning for MAS Information Flow Optimization Input :Active agent set A, Indicator Pool I, Rectifier Model Φrect, Embedding Model Memb Parameters:Max iterations Tmax, Top-K retrieval Kact, Safety threshold γ Output :Final answer Y

// Phase 1: Agent Execution & Rectification

- 1 O ← ∅ ; // Initialize valid output set
- 2 foreach Agent Ai ∈ A do

- 3 o(0)i ← Φi(xi,Ri,Ki) ; // Initial Generation
- 4 t ← 1;
- 5 while t ≤ Tmax do

- // Step 1: Two-Stage Indicator Retrieval

6 Sscen(t) ,Sact(t) ← Φrect(o(it)) ; // Extract keywords 7 q(it) ← Memb(Sscen(t) ⊕ Sact(t)) ; // Compute query 8 Icand(t) ← Top-Kcand(q(it),I) ; // Coarse: Semantic matching 9 Iact(t) ← Φrect(Icand(t) | xi,Ri,o(it)) ; // Fine: Select ≤ Kact active indicators

- // Step 2: Verification

10 E(t) ← 0, F(t) ← ∅; 11 foreach Indicator Ik ∈ Iact(t) do 12 (vk(t),rk(t)) ← Φrect(o(it) | xi,Ri,Ik); 13 if vk(t) = 1 then 14 E(t) ← 1; 15 F(t) ← F(t) ∪ {rk(t)};

- // Step 3: Tri-State Gating Decision

- 16 p(t) ← 1 − |F(t)|/|Iact(t)| ; // Calculate proportion of passed indicators
- 17 if p(t) ≥ τpass then

- 18 oi ← o(it);
- 19 O ← O ∪ {oi};
- 20 break ; // Pass: Accept output

- 21 else if t < Tmax then

- 22 o(it+1) ← Φi(xi,Ri,Ki,F(t)) ; // Retry: Regenerate
- 23 t ← t + 1;

- 24 else

- 25 oi ← ∅ ; // Reject: Discard output
- 26 break; // Propagate Output to Successors

- 27 if oi ̸= ∅ then

- 28 foreach Agent Aj ∈ N(Ai) do

- 29 Kj ← {Ri,oi}

// Phase 2: Global Fallback Check

- 30 Nvalid ← |{o ∈ O | o ̸= ∅}|;
- 31 if Nvalid < γ then

- 32 Trigger System-Wide Reset;
- 33 Discard O and re-initialize with fresh agents;

- 34 return oN;

Algorithm 2: Failure-Driven Indicator Pool Construction Input :Source Dataset Dsrc = {Q,Y∗}, Teacher Model Φteach, Deduplication Model Φdedup,

Embedding Model Memb Parameters:Retrieval size Kdedup Output :Optimized Indicator Pool I

// Initialize empty indicator pool

- 1 I ← ∅;
- 2 foreach Instance (Q,Y∗) ∈ Dsrc do // Step 1: Failure Trajectory Collection

- 3 Execute MAS to obtain trajectory: T = (Q,A1:N,o1:N,Y);
- 4 if Y ̸= Y∗ then // Step 2: Offline Indicator Mining

- 5 foreach Agent Ai in T do

- 6 Inew ← Φteach (T ,Y∗,Ri,oi) ; // Generate candidate indicators // Step 3: Redundancy Elimination
- 7 foreach Indicator Inew = (nnew,dnew,cnew) ∈ Inew do

- 8 vnew ← Memb(dnew ⊕ cnew) ; // Compute semantic vector
- 9 Isim ← Top-Kdedup(vnew,I) ; // Retrieve top-Kdedup similar set
- 10 IsNovel ← Φdedup (Inew,Isim);
- 11 if IsNovel is True then

- 12 I ← I ∪ {Inew} ; // Add novel error pattern

- 13 return I;

An Indicator Example

Name: SQUARE_ROOT_MANIPULATION_CHECK: Detailed Definition: This error occurs when mathematical expressions in radical form are manipulated in a way that leads to logical contradictions relating to divisibility or integer properties. Trigger Condition: When the agent performs manipulations involving expressions in radical form and calculations involving divisibility and integer properties.

Figure 6: An example of the indicators from the constructed pool for the math domain.

General Indicator for Math

Name: CRITICAL_MATH_LOGIC_AUDIT Detailed Definition: A focused audit to detect substantive logical fallacies, calculation errors, or conditional oversights that invalidate the final result. Trigger Condition: The Agent is performing mathematical reasoning, derivation, or calculation.

Figure 7: The design of the general indicator for the math domain.

General Indicator for Code

Name: CRITICAL_CODE_CORRECTNESS_CHECK Detailed Definition: A functional audit focusing on runtime safety, logical integrity, and adherence to requirements in code implementation. Trigger Condition: The Agent is generating, debugging, or analyzing computer code.

Figure 8: The design of the general indicator for code domain.

Prompt for Math Rectifier

You are an Objective Logic Auditor. Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL LOGIC ERROR** regarding a specific **Area of Concern**.

### The "Impact & Action" Protocol

- 1. **Presumption of Validity**: You must assume the Agent’s reasoning is correct unless you find irrefutable evidence of a fatal flaw.
- 2. **The "Actionability" Test**: If you cannot provide a specific, mathematical correction (a formula, a step, or a value), **IT IS NOT A FLAW**.
- 3. **The "Impact" Test**: If the Agent’s phrasing is imperfect but the **FINAL ANSWER** remains mathematically correct, **IT IS NOT A FLAW**.

### Judgment Criteria

**[Area of Concern]**: {trigger_condition}

### CONTEXT

- - **Task**: {task}
- - **Agent Role**: {role}
- - **Agent Output**: {agent_output}

—

### OUTPUT FORMAT (JSON ONLY) You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{ "evidence_quote": "Verbatim quote of the problematic part. Write ’N/A’ if valid.", "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on logic, not style. Try to express in a concise and to the point manner, avoid lengthy speeches. Write ’N/A’ if valid.", "suggestion": "Concrete instruction on how to fix it (e.g., ’Change x to y’, ’Apply formula Z’). If no fix is needed or possible, write ’N/A’.", "impact_assessment": "Simulate the correction. Does the FINAL ANSWER or core conclusion change? (YES/NO) and brief reason.", "is_flawed": boolean // Set to true ONLY if ’suggestion’ is concrete AND ’impact_assessment’ is YES. Otherwise false. }

Figure 9: The prompt template for math rectifiers.

Prompt for Code Rectifier

You are a Senior Code Auditor and Architect. Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL CODING ERROR** regarding a specific **Area of Concern**.

### The "Impact & Action" Protocol 1. **Presumption of Validity**: You must assume the Agent’s code is functionally correct unless you find irrefutable evidence of a fatal flaw (syntax error, logic bug, or interface violation).

- 2. **The "Actionability" Test**: If you cannot provide a specific code correction (a line change, a logic fix, or a parameter adjustment), **IT IS NOT A FLAW**.
- 3. **The "Impact" Test**: If the code is inefficient, verbose, or stylistically non-standard but **EXECUTES CORRECTLY** and returns the right result, **IT IS NOT A FLAW**.

### Judgment Criteria

**[Area of Concern]**: {trigger_condition}

### CONTEXT

- - **Task**: {task}
- - **Agent Role**: {role}
- - **Agent Output**: {agent_output}

—

### OUTPUT FORMAT (JSON ONLY) You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{ "evidence_quote": "Verbatim quote of the problematic code snippet. Write ’N/A’ if valid.", "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on functional correctness (bugs/crashes), not style (PEP8/comments).Try to express in a concise and to the point manner, avoid lengthy speeches. Write ’N/A’ if valid.", "suggestion": "Concrete instruction on how to fix the code (e.g., ’Change index i to i+1’, ’Import module X’). If no fix is needed, write ’N/A’.", "impact_assessment": "Simulate the correction. Does it fix a runtime error, infinite loop, or incorrect output? (YES/NO) and brief reason.", "is_flawed": boolean // Set to true ONLY if ’suggestion’ is concrete AND ’impact_assessment’ is YES. Otherwise false. }

Figure 10: The prompt template for code rectifiers.

###### Prompt for Teacher

You are an AI acting as a **Lead Mathematics Auditor and Logic Specialist**, specifically optimized for the MATH dataset (high-difficulty competitions like AMC, AIME). ### Background & Goal

**Background**: An agent team has attempted to solve a complex math problem, and **the team’s final answer is INCORRECT**.

**Goal**: Synthesize the known problem (‘problem‘), standard solution (‘solution‘), and the Agent’s ‘output‘ to strictly evaluate the Agent’s reasoning process.

**IMPORTANT CONTEXT**: The provided ‘solution‘ is a standard, single-path reference answer. However, the Agent is part of a Multi-Agent System (MAS).

- - Its ‘output‘ depends on its ‘agent_role‘ (e.g., a "Python Coder" writes code, a "Critic" critiques).
- - **DO NOT** penalize the Agent simply because its output does not look like the standard ‘solution‘ (e.g., using code instead of pure derivation is valid if the role permits).
- - Only penalize **logical errors**, **calculation errors**, or **hallucinations** that contradict mathematical truths. ### MATH Input Context

- 1. **‘problem‘**: {problem}
- 2. **‘solution‘**: (Ground Truth) {solution}
- 3. **‘agent_role‘**: {agent_role}
- 4. **‘output‘**: (Agent’s Attempt) {output}

- ### Phase 1: Diagnosis Please execute the following logical judgment:

- 1. Assess whether the Agent’s output is logically and mathematically correct **within the scope of its role**.
- 2. **AUDIT STRATEGY (CRITICAL)**:

- - **DO NOT STOP at the first error.** You must scan the ENTIRE output line by line.
- - Independent errors often exist (e.g., a logical fallacy in Step 1 AND a formatting error in the Final Answer).
- - You are expected to find **MULTIPLE distinct errors** (less than 5) if they exist.

- 3. **Decision**:

- - If the output contains **NO errors**: Output ‘NO_ERROR‘.
- - If the output contains **errors**: Identify **ALL** of them and proceed to Phase 2.

- ### Phase 2: Metric Extraction Transform **EACH identified error** separately into a **generalized** JSON metric object.

**CRITICAL**: The ‘name‘, ‘detailed_definition‘, ‘trigger_condition‘, and ‘example_error‘ must be **generalizable** to other similar math problems.

- 1. **‘name‘**:

- * **Requirement**: Summarize the error pattern. It can be **appropriately longer** to avoid ID collisions.
- * **Format**: ‘UPPER_CASE_WITH_UNDERSCORES‘.

- 2. **‘domain_tag‘**:

- * **Requirement**: Classify this error into a specific mathematical or operational domain.
- * **Examples**: "Geometry", "Probability", "Algebra", "Number Theory", "Python Implementation", "Logical Reasoning".

- 3. **‘detailed_definition‘**:

- * **Requirement**: Define the **ROOT CAUSE** or **Mental Misconception** behind the error. Do not just say "used wrong formula"; explain "confused concept A with concept B".
- * **Format**: "This error occurs when the agent [misconception], leading to [consequence]."

- 4. **‘evaluator_prompt‘**: Contains the trigger condition for retrieving this metric:

- * **‘trigger_condition‘**:
- * **Requirement**: Describe the **Context** or **Action** where this error is likely to happen. **DO NOT** assume the error has already occurred (Decriminalized).
- * **Format**: "When the problem involves [context]..." OR "When the agent attempts to [action]..."

- 5. **‘example_error‘**:

- * **Requirement**: Provide a concrete example of the error AND the logic for why it is wrong/how to fix it.
- * **Format**: "Error Snippet: [Quote agent’s wrong step] | Correction Logic: [Explain why it is wrong and what the correct approach/formula should be]." ### Output Format

- - If no error: Output ‘NO_ERROR‘ only.
- - If errors exist: **ALWAYS Output a JSON LIST** containing one or more metric objects.
- - Structure: ‘[ { "name": "ERROR_1", ... }, { "name": "ERROR_2", ... } ]‘
- - **CRITICAL JSON SYNTAX RULE**:
- - When writing LaTeX inside JSON strings, **YOU MUST DOUBLE-ESCAPE BACKSLASHES**.
- - **WRONG**: ‘"equation": " frac{1}{2}"‘ (This causes JSON parse error!)
- - **CORRECT**: ‘"equation": " frac{1}{2}"‘ (This works!)

Figure 11: The prompt template for the teacher model during indicator pool construction.

Table 5: Dataset statistics

Domain Dataset Size Test Set

GSM8K (Cobbe et al., 2021) 1,319 MATH-500 (Lightman et al., 2024) 500 AQuA (Patel et al., 2021) 254 AMC23 40 OlympiadBench (He et al., 2024) 675 OlymMATH Easy (Sun et al., 2025) 100 OlymMATH Hard (Sun et al., 2025) 100

Math

- AIME24 (Zhang and Math-AI, 2024) 30
- AIME25 (Zhang and Math-AI, 2025) 30

MBPP (Austin et al., 2021) 257 HumanEval (Chen et al., 2021) 161 CodeContests (Li et al., 2022) 165 LiveCodeBenchV1 (Jain et al., 2025) 400

Code

Training Set Math

MATH 2,000 AQuA 2,000

MBPP 120 KodCode (Xu et al., 2025) 10,000 CodeContests 1,000

Code

Indicator Pool Math - 2,000 Code - 2,545

[Figure 103]

Math Task Q and Correct Answer Y∗

Task: Let S be the union of the set of all points inside a regular nonagon with side length 2 units and the set of all points less than 1 unit away from a point on the perimeter of the nonagon. What, in units, is the perimeter of S?

Correct Answer:

|18 + 2π|
|---|

Reference logic: The offset boundary keeps nine straight portions of length 2, contributing 18. The rounded pieces around the nine vertices have total turning angle 2π, and radius 1, contributing 2π. Hence the perimeter is 18 + 2π.

Figure 12: An example of the given math task.

[Figure 104]

Math Solver: Initial Output o(0)

The solver is asked to verify two candidate answers for the nonagon buffer perimeter:

- - Participant_1:

|18 +<br><br>9 2<br><br>π|
|---|

, from treating the nine rounded corners as quarter-circles.

- - Participant_2:

|18 + 7π|
|---|

, from using the regular nonagon’s internal angle for the nine rounded-corner arcs. It first computes the original nonagon perimeter:

##### 9 × 2 = 18.

It then rejects the quarter-circle interpretation and accepts the internal-angle interpretation. The solver states that the internal angle of a regular nonagon is

(9 − 2)π 9

7π 9

. Using radius 1, it calculates each corner arc as

=

7π 9

7π 9

1 ·

, so the nine arcs contribute

=

7π 9

9 ·

= 7π. The straight portions contribute 18. The solver therefore outputs

|18 + 7π|
|---|

.

Figure 13: The initial output of the math solver agent.

[Figure 105]

Rectifier: Rectification Round 0/2 Evaluation

- [1] Matched Indicators:

- - PERIMETER_CALCULATION_MISCOMPONENTS
- - PERIMETER_PROPAGATION_ERROR
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK
- - INCORRECT_PERIMETER_FORMULA
- - ANGLE_DEPENDENCY_CHECK

- [2] Evaluations:

- PERIMETER_CALCULATION_MISCOMPONENTS: Flawed

Evidence: Participant_1 claims 18 + 92π. Analysis: The quarter-circle interpretation uses a fixed 90◦ arc at each vertex, which does not match the geometry

of a regular nonagon. Suggestion: Replace the quarter-circle assumption with arcs tied to the nonagon’s angle.

- - PERIMETER_PROPAGATION_ERROR: Flawed Evidence: The solver discusses rounded corners and the propagation from local corner shape to total perimeter. Analysis: The local arc angle is propagated into the final perimeter formula. Suggestion: Re-evaluate the angle used for every rounded corner.
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK: Flawed Evidence: The response compares a quarter-circle calculation with an internal-angle calculation. Analysis: The calculation combines straight side contributions and circular arc contributions, but the corner-angle

component is under-audited. Suggestion: Recompute the corner sector before summing with the straight contribution.

- INCORRECT_PERIMETER_FORMULA: Flawed Evidence: The final formula adds 18 and 7π. Analysis: The formula inherits the chosen arc angle directly, so an incorrect angle selection changes the final

perimeter. Suggestion: Rebuild the perimeter formula from the offset-boundary geometry.

- ANGLE_DEPENDENCY_CHECK: Flawed Evidence: The solver chooses between 90◦ and 7π/9. Analysis: The answer depends on choosing the correct angle type for the arc. Suggestion: Audit whether the arc angle should follow the polygon’s internal angle, external angle, or another

derived angle.

- [3] Pass Count: 0/5, threshold 60%, action: retry.
- [4] Feedback Sent to Solver: The auditor flags the quarter-circle assumption and asks the solver to revise the corner arcs using the regular nonagon’s angle. The feedback still points the solver toward the internal-angle explanation, which leaves room for a second wrong answer.

Figure 14: The rectifier’s judgments and feedback to the initial output.

[Figure 106]

Math Solver: Rectification Round 1/2 Output o(1)

The solver accepts the first audit’s feedback. It states that the original quarter-circle assumption was invalid because the internal angle of a regular nonagon is

7π 9

, and then treats this internal angle as the angle of each rounded corner. The solver describes S as a Minkowski-style offset of the nonagon by a disk of radius 1. It then decomposes the boundary into straight portions and rounded corners:

- 1. Internal angle of the nonagon: (9 − 2)π

9

=

7π 9

.

- 2. Arc length for each rounded corner:

1 ·

7π 9

=

7π 9

.

- 3. Total rounded-corner contribution:

9 ·

7π 9

= 7π.

- 4. Straight-side contribution:

9 × 2 = 18. The solver therefore keeps the same final answer:

|18 + 7π|
|---|

.

Figure 15: The output of the math solver agent in Rectification Round 1.

[Figure 107]

Rectifier: Rectification Round 1/2 Evaluation

- [1] Matched Indicators:

- - ANGLE_DEPENDENCY_CHECK
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK
- - CALCULATION_CONTINGENT_ON_ANGLES
- - PERIMETER_PROPAGATION_ERROR
- - IGNORING_STRAIGHT_SEGMENT_CONTRIBUTIONS

- [2] Evaluations:

- - ANGLE_DEPENDENCY_CHECK: Correct The response explicitly recognizes that the perimeter calculation depends on the angle assigned to each rounded corner.
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK: Flawed Evidence: The response uses the nonagon’s internal angle 7π/9 as the rounded-corner arc angle. Analysis: For a unit offset of a polygon, the rounded corner is determined by the exterior turning angle. For a

regular nonagon, this angle is 2π/9.

Suggestion: Use the external angle 2π/9 for each radius-1 arc. Impact: The arc contribution changes from 7π to 2π, so the final perimeter changes from 18 + 7π to 18 + 2π.

- CALCULATION_CONTINGENT_ON_ANGLES: Flawed Evidence: The solver computes 1 · 7π/9 per corner. Analysis: The arithmetic follows from the chosen angle, but the chosen angle type is wrong. The offset boundary

uses the external angle. Suggestion: Compute each arc as 1 · 2π/9 = 2π/9, giving total arc length 2π.

- PERIMETER_PROPAGATION_ERROR: Flawed

Evidence: The final answer 18 + 7π is produced by propagating the internal-angle arc length across nine corners. Analysis: The wrong angle is carried through the perimeter calculation.

Suggestion: Replace the total arc term 7π with 2π.

- IGNORING_STRAIGHT_SEGMENT_CONTRIBUTIONS: Correct The solver retains the straight-side contribution 9 × 2 = 18.

- [3] Pass Count: 2/5, threshold 60%, action: retry.
- [4] Feedback Sent to Solver: An external auditor has reviewed the previous output and flagged the core angle type error. The feedback states that the rounded-corner arc is controlled by the regular nonagon’s external angle, 2π/9, rather than its internal angle, 7π/9. With radius 1, each arc has length 2π/9, and the nine arcs contribute 2π. The corrected target expression is therefore 18 + 2π.

Figure 16: The rectifier’s judgments and feedback to the Round 1 output.

[Figure 108]

Math Solver: Rectification Round 2/2 Output o(2)

The solver accepts the second audit’s feedback and revises the geometry. It now states that the internal angle of the nonagon does not control the rounded corner in the offset boundary. Instead, each corner arc is determined by the external angle of the regular nonagon.

Corrected calculation:

- 1. External angle of the nonagon: 2π

9

.

- 2. Arc length for each rounded corner, with radius 1:

1 ·

2π 9

=

2π 9

.

- 3. Total rounded-corner contribution across all nine vertices:

9 ·

2π 9

= 2π.

- 4. Straight-side contribution:

9 × 2 = 18. Thus the corrected perimeter is

18 + 2π, and the solver outputs

|18 + 2π|
|---|

.

Figure 17: The output of the math solver agent in Rectification Round 2.

[Figure 109]

Rectifier: Rectification Round 2/2 Evaluation

- [1] Matched Indicators:

- - CALCULATION_CONTINGENT_ON_ANGLES
- - INCORRECT_SUMMATION_OF_SEGMENT_LENGTHS
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK
- - ANGLE_DEPENDENCY_CHECK
- - ANGLE_TYPE_IDENTIFICATION

- [2] Evaluations:

- - CALCULATION_CONTINGENT_ON_ANGLES: Correct
- - INCORRECT_SUMMATION_OF_SEGMENT_LENGTHS: Correct
- - GEOMETRIC_COMPOUND_CALCULATION_CHECK: Correct
- - ANGLE_DEPENDENCY_CHECK: Correct
- - ANGLE_TYPE_IDENTIFICATION: Correct

- [3] Pass Count: 5/5, threshold 60%, action: released.
- [4] Feedback: [EMPTY]

Figure 18: The rectifier’s judgments and feedback to the Round 2 output.

