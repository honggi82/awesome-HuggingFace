## RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents

Jialiang Zhu*†1 Gongrui Zhang*†1 Xiaolong Ma*†2 Lin Xu*†3 Miaosen Zhang†1 Ruiqi Yang†4 Song Wang†5 Kai Qiu* 6 Zhirong Wu* 6 Qi Dai6 Ruichun Ma6 Bei Liu6 Yifan Yang6 Chong Luo6 Zhengyuan Yang6 Linjie Li6 Lijuan Wang6 Weizhu Chen6 Xin Geng1 Baining Guo6

# arXiv:2602.02486v1[cs.CL]2Feb2026

### Abstract

LLM-based deep research agents are largely built on the ReAct framework. This linear design makes it difficult to revisit earlier states, branch into alternative search directions, or maintain global awareness under long contexts, often leading to local optima, redundant exploration, and inefficient search. We propose Re-TRAC, an agentic framework that performs cross-trajectory exploration by generating a structured state representation after each trajectory to summarize evidence, uncertainties, failures, and future plans, and conditioning subsequent trajectories on this state representation. This enables iterative reflection and globally informed planning, reframing research as a progressive process. Empirical results show that Re-TRAC consistently outperforms ReAct by 15–20% on BrowseComp with frontier LLMs. For smaller models, we introduce ReTRAC-aware supervised fine-tuning, achieving state-of-the-art performance at comparable scales. Notably, Re-TRAC shows a monotonic reduction in tool calls and token usage across rounds, indicating progressively targeted exploration driven by cross-trajectory reflection rather than redundant search. Code and models are available at GitHub link.

### 1. Introduction

Large language models (LLMs) have progressed from single-turn question answering to chain-of-thought reasoning (Wei et al., 2022), function calling (Schick et al., 2023), and complex multi-turn agentic applications (Anthropic,

*Equal Core Contributors †This work was done during the internship at MSRA 1Southeast University 2Waseda University 3Tsinghua University 4Brown University 5Zhejiang University 6Microsoft. Correspondence to: Kai Qiu <Kai.Qiu@microsoft.com>, Zhirong Wu <Wu.Zhirong@microsoft.com>.

Preprint. February 3, 2026.

2025). This evolution reflects a shift from passive response generation toward autonomous, goal-directed problem solving in open environments. A deep research agent (OpenAI, 2025a; Google, 2025), capable of autonomously searching the open web and gathering and analyzing information from thousands of web pages, represents the next frontier of information retrieval for general intelligence.

Most existing deep research agents are built upon the ReAct paradigm (Yao et al., 2022), which interleaves large language model (LLM) reasoning steps with tool invocation, appending both into the model context in a linear sequential manner. In this work, we provide an in-depth analysis of the inherent limitations of ReAct-style linear reasoning workflows. Although LLM reasoning can be trained to support behaviors such as backtracking and self-reflection (Guo et al., 2025), this strictly linear agentic workflow is not well suited for open-ended tasks that require broad exploratory investigation. Revisiting earlier reasoning states and branching into alternative search trajectories remains challenging, particularly under long-context settings (e.g., 128K–256K tokens), where context management and credit assignment become increasingly difficult. Consequently, the ReAct framework is susceptible to issues such as local optima, redundant exploration, and inefficient search dynamics (Yao et al., 2023).

To empower LLM-based agents with diverse exploration capabilities, we propose to explicitly guide agents toward search trajectories that have not been previously explored. This direction is motivated by two key observations. First, existing deep research models (even after extensive reinforcement learning post-training) exhibit substantially higher pass@k performance than pass@1. This gap indicates that repeated inference induces diverse reasoning trajectories, suggesting that model limitations often stem from insufficient exploration within a single trajectory rather than inadequate reasoning capacity. Second, prior work shows that LLMs are generally better at verifying candidate solutions than generating them from scratch (Weng et al., 2023; Singhi et al., 2025), motivating a search paradigm that emphasizes broad candidate generation followed by verification-driven selection.

###### 58.1 57.3

60

60

| |53 49.7| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |44 43.4<br><br>37.3| | | | | | | | | | | |
| |35.3<br><br>30<br><br>25 22.4<br><br>15.7| | | | | | | | | | | |
| | | | | | | | | | | | | |
| |15.3| | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |48.5| | | | | | | | | | | |
| |46.7 45.2 44.1| | | | | | | | | | | |
| |36.1<br><br>32| | | | | | | | | | | |
| | | | | | | | | | | | | |
| |29.2 29 28.4| | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

55

50

50

40

45

Score

Score

30

40

35

20

30

10

25

0

20

RE-TRAC-30BA3B(Ours)TongyiDeepResearch-30BMiniMaxM2-229Bo3IterResearch-30BA3BWebSailorV2-30BA3BRE-TRAC-4B(Ours)AgentCPMExplore-4BNestBrowse-4BWebExplorer-8BInfoAgent-14B

RE-TRAC-30BA3B(Ours)TongyiDeepResearch-30BMiniMaxM2-229Bo3 IterResearch-30BA3BWebSailorV2-30BA3BRE-TRAC-4B(Ours)WebExplorer-8BInfoAgent-14BAgentCPMExplore-4BNestBrowse-4B

BrowseComp

BrowseComp-zh

###### 78.2

80

90

| |75.7 74.1 72.8| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| |70.9 70.5 70.4 68.9| | | | | | | | | | |
| |63.9| | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| |50| | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| |83<br><br>76.6| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| |75 74 73.7 72 70 66.7| | | | | | | | | | |
| |53.7| | | | | | | | | | |
| | | | | | | | | | | | |
| |40.4| | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

75

80

70

70

65

Score

Score

60

60

50

55

40

50

45

30

RE-TRAC-30BA3B(Ours)MiniMaxM2-229BWebSailorV2-30BA3BIterResearch-30BA3BTongyiDeepResearch-30BRE-TRAC-4B(Ours)o3NestBrowse-4BAgentCPMExplore-4BWebExplorer-8B

RE-TRAC-30BA3B(Ours)RE-TRAC-4B(Ours)TongyiDeepResearch-30BNestBrowse-4BWebSailorV2-30BA3BMiniMaxM2-229BAgentCPMExplore-4BWebExplorer-8Bo3InfoAgent-14B

GAIA

XBench

- Figure 1. Comparison of RE-TRAC with state-of-the-art agentic models. Our 4B and 30B models surpass the performance of significantly larger, state-of-the-art models.

We propose Re-TRAC, an agentic framework that recursively constructs structured state representation at the end of each trajectory and uses them as the prompting context for subsequent trajectories. Each state representation summarizes the evolving state of investigation along multiple dimensions, including accumulated evidence, unresolved uncertainties, identified failure modes, and a forward-looking research plan. Unlike multiple independent trajectories that operate in isolation, Re-TRAC enables iterative reflection, cross-trajectory knowledge consolidation, and globally informed planning. This design transforms exploration from a set of disconnected attempts into a progressively informed search process. Empirically, we observe that Re-TRAC agents issue fewer tool calls and consume fewer tokens with each successive round of research, indicating improved decision-making efficiency and more targeted information acquisition guided by prior experience.

Our experiments demonstrate that Re-TRAC achieves absolute gains of 15–20% over ReAct on the BrowseComp benchmark when applied with frontier LLMs. This inspires us to push the limits of abilities of small models via ReTRAC. To unlock the benefits of Re-TRAC for smaller models, we develop a post-training recipe that constructs supervised fine-tuning (SFT) data consisting of trajectories explicitly conditioned on structured state representations. This training procedure teaches the model to ground its reasoning, planning, and tool use on structured cross-trajectory summaries rather than relying solely on immediate context. After fine-tuning, our 30B model achieves 53% accuracy on BrowseComp, while the 4B model reaches 30%, establishing state-of-the-art performance among models of

comparable scale (see Figure 1).

### 2. Related Work

#### 2.1. Deep Research Agents

The emergence of Deep Research Agents marks a transition from simple information retrieval to autonomous systems capable of long-horizon reasoning, strategic planning, and persistent tool utilization (OpenAI, 2025a; Google, 2025; xAI, 2025; Liu et al., 2025a; Perplexity, 2025; MiniMax, 2025; Zhang et al., 2025; Team et al., 2025b; Team, 2025a). Agents powered by proprietary models, such as OpenAI Deep Research (OpenAI, 2025a), Gemini Deep Research (Google, 2025), Claude (Antropic, 2025), Perplexity (Perplexity, 2025), and Grok (xAI, 2025), leverage large-scale training and deep tool integration to achieve high accuracy. In parallel, open-source models, including DeepSeek (Liu et al., 2025a), GLM (Team, 2025a), Kimi (Team et al., 2025a), MiniMax (MiniMax, 2025), and Tongyi Deep Research (Team et al., 2025b), have strengthened their longhorizon capabilities through specialized training on extensive agentic tasks. Additionally, works like InfoAgent (Zhang et al., 2025), WebSailor (Li et al., 2025b), and DeepDive (Lu et al., 2025) have explored foundational challenges such as data synthesis and search-oriented environment construction. Our work introduces a recursive experience compression mechanism to enhance the agent’s ability to handle long-horizon tasks.

#### 2.2. Agentic Context Management

The ability to manage context effectively is critical for agents performing long-horizon tasks. Recent research generally falls into two categories: intrinsic context optimization (Liu et al., 2025a; Team, 2025a) and external memory mechanisms for state maintenance (Chen et al., 2025a; Yu et al., 2025; Wu et al., 2025). For the first category, many agentic LLMs, such as DeepSeek-V3.2 (Liu et al., 2025a), and GLM-4.7 (Team, 2025a), integrate context pruning directly into the agent’s reasoning loop, focusing on compressing observation spaces and pruning redundant trajectory history. Parallel to context pruning, recent works have focused on leveraging external memory. IterResearch (Chen et al.,

- 2025a) and MemAgent (Yu et al., 2025) utilize dynamic memory structures to reconstruct task status at each step, discarding generic history to simulate infinite horizons. ReSum (Wu et al., 2025) introduces a “summarize-and-reset” paradigm, periodically condensing exploration history into compact memory. While our work naturally extends the effective context length to infinity, our primary objective is to critique its own trajectory, engage self-reflection and reinforce correct reasoning paths.

#### 2.3. Test-Time Scaling

While traditional scaling laws have focused on increasing model parameters and training data, recent paradigms have shifted towards test-time compute scaling (Wei et al., 2022; OpenAI, 2025b; Liu et al., 2025a; Du et al., 2023; Wang et al., 2022). The dominant approach to test-time scaling involves expanding the model’s internal reasoning process. Chain-of-Thought (CoT) extensions (Wei et al., 2022) and reasoning models like OpenAI-o3 (OpenAI, 2025b) and DeepSeek-R1 (Guo et al., 2025) incentivize extended internal traces to decompose problems. A parallel direction scales compute via ensemble strategies and inter-agent verification. Self-Consistency effectively marginalizes out reasoning errors by sampling diverse reasoning paths and applying majority voting to select the most robust answer (Wang et al., 2022). Multi-Agent Debate enables separate LLM instances to critique and refine each other’s responses, leveraging adversarial dynamics to improve factuality and reduce hallucinations (Du et al., 2023). Our work introduces a sequential dimension to test-time scaling that differs from the parallel nature of voting or debate. We devise a novel mechanism to catalyze continuous self-reflection, enabling the model to explore a broader spectrum of possibilities with high computational efficiency.

### 3. Motivation

Through a systematic analysis of LLMs in Deep Research tasks, we identify two fundamental limitations that hinder performance. First, current models suffer from insufficient

exploration, often converging prematurely on sub-optimal reasoning paths. While a naive solution to encourage exploration is to allow multiple trials (e.g., majority voting or Best-of-N), this introduces a secondary challenge: informational efficiency. The core problem lies in how to efficiently leverage these diverse trajectories to synthesize a superior final output.

Incomplete Branch Exploration. In order to find the bottleneck of current advanced deep research agents, we collect and analyze their trajectories where they fail to output correct answers. The analysis reveals a common phenomenon: in most failed trajectories, there are branches that the model plans to explore but forgets to explore in the end. As shown in Table 1, the ratio of this case can be up to 93%. We attribute this pervasive under-exploration to the fundamental structural mismatch between the long-horizon nature of deep research tasks and the inherent linearity of the ReAct framework. While deep research necessitates strategic branching and backtracking, the ReAct paradigm constrains the agent to a sequential execution path, creating a discrepancy that inhibits the model’s ability to pivot or re-evaluate earlier decisions. Deep research tasks typically demand extended trajectories, which often spans hundreds of thousands of tokens, characterized by a high density of interdependent tool calls. We observe that within the constraints of the linear ReAct framework, LLMs exhibit catastrophic forgetting as the trajectory lengthens. This is primarily because the model struggles to maintain long-term planning coherence. The critical task-level objectives formulated in the early stages are often marginalized by the accumulating volume of intermediate tool calls and observations.

Table 1. Ratio of trajectories containing incomplete branches among all failed trajectories. Evaluated on BrowseComp (Wei et al., 2025). Details are in Section A.

Model GLM-4.7 DeepSeek-V3.2 Tongyi-DeepResearch Ratio 93.0% 92.7% 83.4%

The Potential from Multiple Trials. A straightforward approach to encourage exploration is through multiple stochastic trials (Wei et al., 2025). To quantify the untapped potential of extensive exploration, we evaluate various LLMs using the Pass@K metric. As illustrated in Figure 2, the substantial gap between Pass@1 and Pass@8 performance reveals a significant performance ceiling that current models fail to reach in a single trajectory.

Our empirical observations suggest that many failures are not rooted in the inherent reasoning capabilities of LLMs, but rather in the absence of an effective exploration management mechanism. While existing paradigms such as Majority Voting and Best-of-N (Wei et al., 2025) allow for multiple trials, these attempts remain independent. This

80

70

###### Accuracy(%)

60

50

- o3

GLM-4.7

DeepSeek-V3.2

GPT-5-medium

- o4-mini

40

30

1 2 3 4 5 6 7 8

Pass @ K (K = 1 to 8)

- Figure 2. Performance of Pass@8 serves as the theoretical upper bound of the models.

lack of inter-trajectory communication leads to two critical inefficiencies: first, it results in repeated and redundant exploration, which squanders computational resources; second, it precludes the possibility of cross-trajectory experience sharing, making it difficult for the model to synthesize a global optimum from isolated experiences. This motivates a trajectory-level recursive agent framework. Instead of starting each attempt from scratch, the model explicitly compresses previous trajectories into a comprehensive experience of verified information and a meticulous enumeration of incomplete branches. By incorporating this feedback into K sequential executions, it may systematically solve the planning and context issues identified in our analysis.

- 4. Method: Re-TRAC Framework

Re-TRAC (Recursive TRAjectory Compression), is an iterative trajectory-level framework. It utilizes a standardized compression specification to summarize previous attempts and propagates this context across successive rollouts. This mechanism ensures that each rollout is both efficient and informed by previous experiences. By continuously expanding the known search space, Re-TRAC effectively broadens plan coverage, reduces redundant exploration, and mitigates dead-end traps.

- 4.1. Trajectory Compression as a Structured State Representation

- Figure 3 contrasts the standard ReAct paradigm (left) with our Re-TRAC framework (right). In ReAct, each rollout is a linear chain starting from the original query. Long contexts induce “incomplete branch exploration”: as token count increases, early plans become less actionable, and the agent often loses track of decisive cues embedded in earlier observations. As illustrated in the left example, the agent may enumerate several candidate branches but fail to follow through, resulting in incomplete exploration coverage.

Re-TRAC solve these issues through trajectory compression (see Figure 4). After each rollout t, the trajectory τt is

###### Help me find info about X

User A, B, C

Search space

[Figure 1]

[Figure 2]

Search space

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

X=1

No!

User A

- Rollout 1 Search space

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

User B

X=2

No!

- Rollout 2

User C

X=3

Nice!

- Rollout 3

Search space

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

User A, B, C

Help me find info about X

[Figure 45]

: “Now I have ……, I still need to search ……”

- Round 1

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

: “Now I have ……, I still need to search ……”

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

: “Now I have ……, All info found, let’s response.

- Round 2
- Round 3

X=3

Nice!

User A, B, C

Figure 3. ReAct versus RE-TRAC framework. ReAct leads to premature convergence and forgotten branches in long-horizon tasks (left). RE-TRAC compresses experience from previous rollouts to systematically guide exploration in successive rounds (right).

distilled into a structured state representation St. Following a fixed compression specification C, the state is iteratively updated:

St ← Compress(τt,St−1;C). (1)

For deep research tasks, we define St through three complementary facets that provide a comprehensive state representation for the agent:

- • Answer & Analytical Conclusions: This facet records the best-supported partial answers. It also stores key inferences from the trajectory. Intermediate conclusions are kept as reusable anchors for later reasoning.
- • Evidence Base & Source Verification: This facet records observed evidence and its provenance. It tracks which sources were consulted. It also marks which claims were verified. This helps avoid redundant tool calls and repeated checking.
- • Uncertainties & Exploration Trace: This facet records what remains unresolved. It includes open hypotheses and candidate branches. It also logs failed attempts and discarded directions. It helps the model to find unexplored search-space for the next rollout.

This structured state is added to the input of the subsequent rollout, ensuring that the agent starts each new attempt with a clear understanding of what is verified, what remains unresolved, and where to focus its exploration.

- 4.2. Recursive Execution with Structured State Representation

ReAct + Pass@K RE-TRAC

Re-TRAC is recursive by design. This process can be sustained over multiple rounds. The initial rollout functions

[Figure 94]

###### RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents

ReAct Pass @ N

Final Answer

###### Tool Result

###### Reasoning & Action

###### Tool Result

Final Answer

###### Reasoning & Action

Question I should check states with state follower , , ,

I should check

I checked clues …. ,

states with state follower , , ,

I checked clues …. , I also checked states with state follower

states with state follower are :

[Figure 95]

[Figure 96]

[Figure 97]

I also checked states with state follower

[Figure 98]

[Figure 99]

[Figure 100]

states with state

Two public elementary schools served the following lunch menus during a week in February (1970s): … and …. They are in a state whose official flower is shared with another

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

follower are :

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

, , ,

Georgia, Iowa, New York, ...

Search: states with state follower

[Figure 109]

[Figure 110]

[Figure 111]

Georgia, Iowa, New York, ...

Search: states with state follower

, ,

[Figure 112]

[Figure 113]

… so, the answer is …

… so, the answer is …

Q … Ans: XYZ School

Q … Ans: ABC School

state. What is the first

[Figure 114]

[Figure 115]

school’s name?

Wrong Anwer

Rollout 1 Rollout 2

###### Correct Answer

RE-TRAC

###### Compressed traj.

Compress specification

Final Answer Up to now I have many observation: clue1… clue2 … But is unsolved

- 1) Answer & Analytical Conclusions
- 2) Evidence Base & Source Verification
- 3) Uncertainties & Exploration Trace

###### Tool Result

###### Final Answer

###### Reasoning & Action

Reasoning & Action

I should check states with state follower , , ,

I checked clues …. , I also checked states with state follower

I should check

I checked clues …. , I also checked states

states with state

[Figure 116]

[Figure 117]

[Figure 118]

states with state follower

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

follower are :

with state follower

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Georgia, Iowa, New York, ...

, , ,

Search: states with state follower

Search: states with state follower

, ,

[Figure 130]

[Figure 131]

… so, the answer is …

… so, the answer is …

if Round<N

…

Q … Ans: likely to be ABC

###### Compress

Ans: XYZ School

[Figure 132]

Round=1

Not Judge

if Round==N

Recursive run Correct Answer

Round+=1

[Figure 133]

Ans: XYZ School

Q Question LLM output Tool Result

- Figure 4. A comparative overview of the independent ReAct Pass@N (top) and our proposed RE-TRAC framework (bottom). Unlike the traditional ReAct paradigm, where multiple rollouts are executed in isolated silos without experience sharing, RE-TRAC is an iterative, trajectory-level framework. It employs a compression mechanism to distill analytical conclusions, evidence, and uncertainties from previous attempts. This compressed context is then propagated to successive rollouts, enabling the agent to recursively reflect on its trajectory and progressively improve its exploration strategy.

like a standard ReAct execution.Consequently, it suffers from the same limitations, such as neglecting early planned branches.

The state representation acts as a guided search update. It filters out low-level trace details that unnecessarily consume context. Crucially, the state prevents exploration from collapsing into a single path. It explicitly retains multiple unresolved candidates and actionable options. This preserves branching diversity for subsequent rollouts. This balance between focused guidance and open branching supports controlled diversity throughout the recursion. Consequently, the agent progressively expands search space coverage while maintaining efficiency.

This recursive approach offers two primary benefits. First, it improves coverage. Incomplete branches are explicitly preserved and executed in later rollouts. Second, it reduces redundancy. The model avoids repeating tool calls for facts that are already verified. A branch missed in the first round is captured in the state and directly explored in the next. In contrast, independent ReAct rounds often waste budget re-exploring the same path. Empirically, this compactness leads to higher efficiency in token usage and tool calls (see Section 5.3).

#### 4.3. Application to Frontier Models

Re-TRAC functions as a training-free prompting strategy. It applies directly to frontier models during inference without

fine-tuning. The execution process is straightforward:

First, we define a Deep Research Query and set a maximum round limit N (defaulting to 8). In the initial round, the model utilizes the standard ReAct framework to generate a complete trajectory. We then compress this trajectory using a prompt (see in Section C.3.1) specifically designed to extract the Structured State Representation. The resulting state is used as the input for the next round. Specifically, it serves as the initial user message, positioned immediately after the system prompt. The model then performs another ReAct cycle to answer the question, building upon the previous state. This process repeats recursively until the round limit N is reached. The answer generated in the final round serves as the final output of Re-TRAC.

In Section 5.3, we benchmark its performance against standard baselines, including Single Run, Best-of-N, Majority Voting, and Weighted Voting.

#### 4.4. Training Small Models for Re-TRAC

Later experiments in Section 5.2 will show that deep research agents of different model sizes, ranging from 229B to 685B, can all benefit from the Re-TRAC workflow. This inspires us to explore whether a tiny edge model can achieve competitive performance, if equipped with the Re-TRAC workflow.

To investigate this, we distill Qwen3-4B-Instruct (Team,

Table 2. Evaluation on deep research benchmarks. Here we evaluate on the BrowseComp full set. Accuracy(%) is reported according to existing studies. Bold indicates the best performance among models with the same size.

Model BrowseComp BrowseComp-zh GAIA XBench HLE

#### Closed-Source Models

Claude-4.5-Sonnet (Anthropic, 2025) 24.1 42.4 71.2 66.0 32 o3 (OpenAI, 2025b) 49.7 58.1 70.5 66.7 24.9 OpenAI DeepResearch (OpenAI, 2025a) 51.5 42.9 67.4 - 26.6 GPT-5-high (Singh et al., 2025) 54.9 63.0 76.7 77.9 42 Gemini-3-pro (DeepMind, 2025) 37.8 51.6 74.8 - 38.3

#### Large Open-Source Models (> 70B)

Kimi-K2-Thinking-1T (moonshotai, 2025) 60.2 62.3 - - 51.0 DeepSeek-V3.2-Thinking-685B (Liu et al., 2025a) 67.6 65.0 - - 40.8 GLM-4.7-358B (zai org, 2025) 52.0 66.6 - - 42.8 MiniMax-M2-229B (MiniMax, 2025) 44.0 48.5 75.7 72.0 31.8

#### Intermediate-sized Open-Source Models (15B∼70B)

Tongyi-DeepResearch-30B-A3B (Team et al., 2025b) 43.4 46.7 70.9 75.0 32.9 IterResearch-30B-A3B (Chen et al., 2025a) 37.3 45.2 72.8 - 28.8 WebSailor-V2-30B-A3B (RL) (Li et al., 2025b) 35.3 44.1 74.1 73.7 30.6 RE-TRAC-30B-A3B (Ours) 53.0 57.3 78.2 83.0 31.5

#### Compact Open-Source Models (< 15B)

InfoAgent-14B (Zhang et al., 2025) 15.3 29.2 - 40.4 WebExplorer-8B (Liu et al., 2025b) 15.7 32.0 50.0 53.7 17.3 AgentCPM-Explore-4B (OpenBMB, 2025) 25.0 29.0 63.9 70.0 19.1 NestBrowse-4B (Li et al., 2025a) 22.4 28.4 68.9 74.0 RE-TRAC-4B (Ours) 30.0 36.1 70.4 76.6 22.2

- 2025b) and Tongyi-DeepResearch-30B-A3B (Team et al., 2025b) from GLM-4.7 on its Re-TRAC trajectories.

To obtain the raw prompts for training, we first construct a large amount of QA pairs via an entity-tree based method, following InfoAgent (Zhang et al., 2025). Specifically, we collect a large batch of entities from WikiPedia as roots of trees. Then for each entity, we recursively search its related entities as child nodes, until the tree grows up to pre-defined depth. The edges between neighbored nodes represents the relationship of the two entities. We synthesize a question by selecting a path from the root to a leaf node and converting the edges into sub-questions . In order to increase the difficulty of the questions, we also fuzzify the sub-questions using o3 (OpenAI, 2025b). Using this pipeline, we construct totally 33K QA pairs. Then, we collect Re-TRAC (4 rounds) trajectories of GLM-4.7 on the synthesized questions, resulting in 104k training samples after filtering, which are used to train our RE-TRAC-4B and RE-TRAC-30B-A3B models via SFT. Details are in Section B.

### 5. Experiments

#### 5.1. Main Results

As shown in Table 2, we evaluate our RE-TRAC models on five challenging search-oriented benchmarks: BrowseComp (Wei et al., 2025), BrowseComp-ZH (Zhou et al., 2025), XBench (Chen et al., 2025b), GAIA (Mialon et al., 2023), and HLE (Phan et al., 2025). A diverse set of competitive models are selected as baselines, which are grouped into four main categories: (1) Closed-Source Models, (2) Large Open-Source Models with more than 70B parameters, (3) Intermediate-sized Open-Source Models with 15B∼70B parameters, (4) Compact Open-Source Models with less than 15B parameters. Evaluation details are illustrated in Section C.

Dominance in the Open-Source Models with the same size. The primary finding from our evaluation is that our RE-TRAC models establish a new state-of-the-art among baselines of the same model size. Our RE-TRAC-30BA3B model consistently achieves 8%∼10% improvement on BrowseComp, BrowseComp-ZH, GAIA and XBench, compared with its base model Tongyi-DeepResearch, which

is also the previous strongest baseline with 30B parameters. For RE-TRAC-4B, it presents the best performance among all the benchmarks, compared with all the baselines with less than 15B parameters. These results demonstrate the superior advantages of our Re-TRAC framework.

Competitive Performance Against Larger Models. Not only achives the best performance among baselines with the same size, Re-TRAC shows competitive ability against those much larger models. Notably, Our 30B model beats MiniMAX-M2-229B on all the benchmarks except HLE. On BrowseComp, its accuracy (53%) also exceeds that of GLM-4.7-358B (52%). This result indicates that the RETRAC framework is able to compensate for the lack of model intelligence by manually expanding its search space.

Exceed Closed-Source Models. While recent opensource large models (Kimi-K2, DeepSeek-V3.2) show a tendency to exceed the advanced closed-source models (GPT5, Gemini-3-pro), our RE-TRAC-30B-A3B maintains this advantage, representing small open-source models. Specifically, it beats all the closed-source baselines on GAIA, and is also the second strongest model on BrowseComp and BrowseComp-ZH. This promising result implies that small models equipped with the Re-TRAC framework can replace those expensive proprietary products, serving as advanced on-device search agents.

#### 5.2. Re-TRAC As a Test-Time Scaling Method

Previous experiment proves that the capacity of small models can be extended by training with Re-TRAC trajectories. In this section, we demonstrate that Re-TRAC can serve as a powerful and efficient training-free test-time scaling method for most general models on deep research tasks.

We implement the Re-TRAC framework for o4-mini (OpenAI, 2025b), o3 (OpenAI, 2025b), GPT-5 (Singh et al., 2025), DeepSeek-V3.2 (Liu et al., 2025a), GLM-4.7 (zai org, 2025) and MiniMax-M2.1 (MiniMax, 2025). Test-time scaling (TTS) methods for comparison include following Browsecomp (OpenAI, 2025a) setting:

- • Re-TRAC (RT@n): Select the final answer of the model after running n RE-TRAC rounds.
- • Majority Voting (MV@n): Select the most frequent answer among n independent solutions.
- • Weighted Voting (WV@n): For n independent solutions, we prompt the model to give a confidence score for its answer and weight each vote by the model’s confidence in that answer. The answer with the most weighted votes are selected.
- • Best-Of-N(Best@n): For n independent solutions, we prompt the model to give a confidence score for its answer. The answer with the highest score are selected.

- Table 3. Performance of different test-time scaling methods on BrowseComp300. Pass@1 is the basic method that the models only have one chance to solve the problem. To ensure fair comparison, all models are evaluated with the common self-hosted search and browse tools, thus the Pass@1 scores can be different with the officially reported scores.

Model Pass@1 RT@8 MV@8 WV@8 Best@8

- o4-mini 25.7 46.8 34.0 46.7 43.3
- o3 54.9 69.8 64.3 69.0 68.0

GPT-5-medium 48.3 66.6 61.7 64.7 54.0 DeepSeek-V3.2 45.3 60.8 55.7 57 55

GLM-4.7 37.7 60.7 41.7 48 42.3

In order to save inference cost, for ablation experiments, we randomly sample 300 questions from BrowseComp as the test cases, consisting BrowseComp300. We empirically find that the model performance on this subset is very close to its performance on the full set.

SOTA Performance. As shown in Table 3, Re-TRAC achieves the best or competitive performance among all the models. Notably, advanced models (o4-mini, o3, GPT-5, DeepSeek-V3.2) can obtain significant gains from all these scaling methods, while GLM-4.7 fails to get comparable improvement via Majority Voting and Best-Of-N. This implies a gap between GLM-4.7 and the other models, in terms of the ability of self-judgment. Under this condition, the fact that all models can benefit from Re-TRAC demonstrates that Re-TRAC is a more general TTS method, and has loose requirements for the intelligence of models.

Economical Spending. Figure 5 exhibits the token usage and tool usage of o3 under different TTS framework. For traditional TTS frameworks, since trajectories are independent of each other, the resource usage increases linearly when scaling up. For Re-TRAC, the model inherits states from previous rounds and its search space converges. Thus, redundant tool calls and exploration can be largely reduced when scaling up. This feature enables Re-TRAC to consume only 50% resources to achieve better performance, compared with other methods.

5.3. Ablations

Effect of SFT Training In previous experiments, we use SFT to train Qwen3-4B-Instruct and Tongyi-DeepResearch on the Re-TRAC trajectories generated by GLM-4.7, producing our RE-TRAC series models.

- Table 4 illustrates the effect of SFT on the performance of our model. Since the 4B model is not pretrained on deep research tasks in large scale, its initial scores on the benchmarks are extremely low. After SFT, it learns basic strategies for searching and utilizing the state representations from previous rounds. Hence, its capability can then be extended by the RE-TRAC framework. This reveals that

RE-TRAC Best-Of-N Majority Voting Weighted Voting

GLM-4.7

DeepSeek-V3.2

###### GPT-5-medium

###### o3

###### o4-mini

70.0

60

60

45

65

67.5

Accuracy(%)

55

40

55

65.0

60

50

62.5

35

50

45

55

60.0

30

40

57.5

45

50

100k 200k 300k

100k 200k 300k Token Usage

50k 100k Token Usage

50k 100k 150k 200k

50k 100k 150k

Token Usage

Token Usage

Token Usage

GLM-4.7

###### DeepSeek-V3.2

###### GPT-5-medium

###### o3

o4-mini

70.0

60

60

45

65

67.5

Accuracy(%)

55

40

55

65.0

60

50

62.5

35

50

45

55

60.0

30

40

57.5

45

50

200 400 600

200 400

50 100 150

100 200 300

100 200

Tool Calls

Tool Calls

Tool Calls

Tool Calls

Tool Calls

- Figure 5. Relationship between model accuracy and used tokens/tools with different TTS methods. Evaluated on BrowseComp300. Re-TRAC consistently achieves better performance with less resources used.

- Table 4. Performance of our model before and after SFT. We evaluate Qwen-3-4B-Instruct using RE-TRAC framework with 8 rounds, the setting of which is the same with the RE-TRAC models. BC and BC-ZH stands for BrowseComp300 and BrowseCompZH.

Model BC BC-ZH GAIA XBench HLE Qwen3-4B-Instruct (RT@8) 2.7 6.9 24.4 45.0 7.0 RE-TRAC-4B 30.0 36.1 70.4 76.6 23.5

we can produce a strong search agent via simple SFT when using Re-TRAC framework, which can achieve comparable or even better performance, compared with those models trained by large-scale reinforcement learning (InfoAgent, WebExplorer in Table 2).

Instructions for Utilizing Summary In our early experiments, we find that the model can be over-relied on the summary of previous rounds and get stuck in previous search path, failing to explore other branches. Hence, we explicitly instruct the model to judge whether the summary is valuable, and expand the search space as much as possible, allowing it to use the summary freely.

- Table 5 shows that with the free-use prompt, the model outperformed in each round compared with the one without the prompt. This result is consistent with our analysis in Section 3 which proposes that the model can get stuck in some search path, failing to explore other possible branches if no other guidance is given.

Quality of state representations In each round of RETRAC, the model adjusts its search space according to the state representations of previous rounds. Hence, the quality of the state representations can affect the final success rate.

To investigate this, we evaluate the performance of RE-

- Table 5. Performance of o3 on BrowseComp300 with and without the prompt instructing the model to freely use the summary. Numbers in the brackets mean accuracy gains compared with previous round.

Round w/o free-use w/ free-use

- 1 56.1 56.1
- 2 61.2 (+5.1) 63 (+7.0)
- 3 64.0 (+2.8) 65.7 (+2.7)
- 4 66.4 (+2.4) 67.0 (+1.3)
- 5 66.8 (+0.4) 69.3 (+2.3)
- 6 68.2 (+1.4) 70.0 (+0.7)
- 7 68.5 (+0.3) 71.0 (+1.0)
- 8 68.9 (+0.4) 71.7 (+0.7)

- Table 6. Effect of Summarizer on the final performance. Evaluated on BrowseComp300 with 8 rounds. Self means use the model itself as summarizer, which is default setting of RE-TRAC.

Model/Summarizer Self GLM-4.7 RE-TRAC-4B 30.0 38.5

RE-TRAC-30B-A3B 53.0 52.4

TRAC models when using GLM-4.7 as the summarizer, as shown in Table 6. The results demonstrates that the 4B model can achieve better accuracy when using a stronger summarizer, while the 30B model has no improvement. This indicates that the summarization ability of our 4B model is relatively weak, and its search abilities have not been fully stimulated. In terms of how to train a search agent with better summarization ability, We leave this to future study.

### 6. Conclusion

We presented Re-TRAC, moving beyond the linear ReAct paradigm toward a recursive, experience-based exploration framework for deep research agents. Re-TRAC facilitates cross-trajectory knowledge consolidation, allowing agents to navigate complex search spaces with higher precision and lower computational overhead. The consistent gains observed in frontier models—and the success of our SFT recipe for smaller models—highlight the importance of structured memory and conditioned planning in autonomous agents. Future work will explore integrating Re-TRAC with reinforcement learning to further optimize the experience generation process and its scalability across even more diverse agentic tasks.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

DeepMind, G. Gemini 3 pro model card, 2025. URL https://storage.googleapis. com/deepmind-media/Model-Cards/ Gemini-3-Pro-Model-Card.pdf.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, 2023.

Google. Gemini deep research — your personal research assistant, 2025. URL https://gemini.google/ overview/deep-research/?hl=en.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Li, B., Wu, J., Yin, W., Li, K., Zhang, Z., Yin, H., Tao, Z., Zhang, L., Xie, P., Zhou, J., and Jiang, Y. Nested browseruse learning for agentic information seeking, 2025a. URL https://arxiv.org/abs/2512.23647.

### References

Anthropic. Introducing claude sonnet 4.5. Anthropic News, September 2025. URL https://www.anthropic. com/news/claude-sonnet-4-5. Accessed: [Insert Access Date].

Antropic. Introducing claude 4, 2025. URL https:// www.anthropic.com/news/claude-4.

Barbaresi, A. Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction. In Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations, pp. 122–131. Association for Computational Linguistics, 2021. URL https://aclanthology.org/2021.

acl-demo.15.

Chen, G., Qiao, Z., Chen, X., Yu, D., Xu, H., Zhao, W. X., Song, R., Yin, W., Yin, H., Zhang, L., et al. Iterresearch: Rethinking long-horizon agents via markovian state reconstruction. arXiv preprint arXiv:2511.07327, 2025a.

Chen, K., Ren, Y., Liu, Y., Hu, X., Tian, H., Xie, T., Liu, F., Zhang, H., Liu, H., Gong, Y., Sun, C., Hou, H., Yang, H., Pan, J., Lou, J., Mao, J., Liu, J., Li, J., Liu, K., Liu, K., Wang, R., Li, R., Niu, T., Zhang, W., Yan, W., Wang, X., Zhang, Y., Hung, Y.-H., Jiang, Y., Liu, Z., Yin, Z., Ma, Z., and Mo, Z. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations, 2025b. URL https://arxiv.org/abs/2506.13651.

Li, K., Zhang, Z., Yin, H., Ye, R., Zhao, Y., Zhang, L., Ou, L., Zhang, D., Wu, X., Wu, J., et al. Websailor-v2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning. arXiv preprint arXiv:2509.13305, 2025b.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Liu, J., Li, Y., Zhang, C., Li, J., Chen, A., et al. Webexplorer: Explore and evolve for training long-horizon web agents, 2025b. URL https://arxiv.org/abs/ 2509.06501.

Lu, R., Hou, Z., Wang, Z., Zhang, H., Liu, X., Li, Y., Feng, S., Tang, J., and Dong, Y. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn rl. arXiv preprint arXiv:2509.10446, 2025.

Mialon, G., Fourrier, C., Wolf, T., LeCun, Y., and Scialom, T. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.

MiniMax. Minimax-m2.1, 2025. URL https: //huggingface.co/MiniMaxAI/MiniMax-M2. 1.

moonshotai. Kimi-k2-thinking, 2025. URL https://huggingface.co/moonshotai/ Kimi-K2-Thinking.

OpenAI. Introducing deep research, 2025a. URL https://openai.com/index/ introducing-deep-research/.

OpenAI. Openai o3 model card, 2025b. URL https://cdn.openai.com/pdf/ 2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf.

OpenBMB. Agentcpm, 2025. URL https://github. com/OpenBMB/AgentCPM.

Perplexity. Introducing perplexity deep research, 2025. URL https://www.perplexity.ai/hub/blog/ introducing-perplexity-deep-research.

Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., et al. Humanity’s last exam, 2025. URL https://arxiv.org/abs/ 2501.14249.

Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., et al. Openai gpt-5 system card, 2025. URL https: //arxiv.org/abs/2601.03267.

Singhi, N., Bansal, H., Hosseini, A., Grover, A., Chang, K.-W., Rohrbach, M., and Rohrbach, A. When to solve, when to verify: Compute-optimal problem solving and generative verification for llm reasoning. arXiv preprint arXiv:2504.01005, 2025.

Team, G. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025a. URL https://arxiv. org/abs/2508.06471.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025a.

Team, Q. Qwen3 technical report, 2025b. URL https: //arxiv.org/abs/2505.09388.

Team, T. D., Li, B., Zhang, B., Zhang, D., Huang, F., Li, G., Chen, G., Yin, H., Wu, J., Zhou, J., et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025b.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wei, J., Sun, Z., Papay, S., McKinney, S., Han, J., Fulford, I., Chung, H. W., Passos, A. T., Fedus, W., and Glaese, A. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

Weng, Y., Zhu, M., Xia, F., Li, B., He, S., Liu, S., Sun, B., Liu, K., and Zhao, J. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 2550–2575, 2023.

Wu, X., Li, K., Zhao, Y., Zhang, L., Ou, L., Yin, H., Zhang, Z., Yu, X., Zhang, D., Jiang, Y., et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025.

xAI. Grok ai deepsearch: Real-time research power guide, 2025. URL https://grokaimodel.com/ deepsearch/.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.-Q., Ma, W.-Y., Liu, J., Wang, M., et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.

zai org. Glm-4.7, 2025. URL https://huggingface. co/zai-org/GLM-4.7.

Zhang, G., Zhu, J., Yang, R., Qiu, K., Zhang, M., Wu, Z., Dai, Q., Liu, B., Luo, C., Yang, Z., et al. Infoagent: Advancing autonomous information-seeking agents. arXiv preprint arXiv:2509.25189, 2025.

Zhou, P., Leon, B., Ying, X., Zhang, C., Shao, Y., Ye, Q., Chong, D., Jin, Z., Xie, C., Cao, M., Gu, Y., Hong, S., Ren, J., Chen, J., Liu, C., and Hua, Y. Browsecompzh: Benchmarking web browsing ability of large language models in chinese, 2025. URL https://arxiv.

org/abs/2504.19314.

### A. Analysis Details of Incomplete Branch Exploration

In Section 3, we find that most deep research agents fail to explore all the proposed branches. This analysis is conducted by collecting their trajectories with incorrect final answers and prompting GPT-5 (Singh et al., 2025) to classify the trajectories. The used prompt is shown in Figure 6.

### B. Training Details

In Section 4.4, we distill Qwen3-4B-Instruct (Team, 2025b) and Tongyi-DeepResearch-30B-A3B (Team et al., 2025b) from GLM-4.7. We use GLM-4.7 to solve 33k questions from the dataset of InfoAgent (Zhang et al., 2025), via our Re-TRAC framework with 4 rounds. Since the context of each round is independent of the others, the solution trajectory for each question can be flattened to 4 training samples. Hence, we obtain 132k raw samples. We drop the samples that satisfy any of the following conditions: 1) The sample contains invalid tool calls. 2) The turn count is less than 15. 3) The sample does not have a valid final answer. This filtering strategy finally results in 104k high-quality training samples. Table 7 lists detailed hyper-parameters.

Table 7. Hyper-parameters for Supervised Fine-Tuning

Setting Value number of samples 104k

learning rate 2e-5

batch size 512 max length 65536 warmup ratio 0.05

learning rate scheduler constant weight decay 0.1

- Adam β1 0.9
- Adam β2 0.95

### C. Evaluation Details C.1. Tools

We implement two primary tools for web-based retrieval and information extraction: search and visit. To ensure fair comparison, we adapt the tool interface to match each model’s native function calling conventions, including parameter names, input types, and output formats.

C.1.1. SEARCH TOOL

The search tool accepts a query string, performs web searches using Google Search Web API, and returns 5 relevant results per query. Each result includes the page title, URL, and a snippet of text. We didn’t implement any addi-

tional processing or filtering on the search results. For Tongyi models, the query parameter accepts a list of strings for batch search queries.

- C.1.2. VISIT TOOL

The visit tool accepts a list of URLs (parameter “urls”, list of strings) and a goal string (parameter “goal”, string), then fetches web pages, extracts their main textual content, and summarizes information relevant to a specified goal. For HTML pages, we use Trafilatura (Barbaresi, 2021) for text extraction. For PDF files, we extract text content page by page. The extracted content is then processed by GPT4o-mini to generate structured summaries with prompt in Figure 7.

For Tongyi models, the parameter is named “url” instead of “urls”, and output includes explicit “Evidence in page” and “Summary” sections.

For GLM models, the tool is named “open” with parameters “url” (single string) and “pattern” instead of “goal”. The output contains only the summary section in a condensed format.

- C.2. Verifier

We follow the evaluation procedure introduced in BrowseComp (Wei et al., 2025) to assess the correctness of model outputs. For each question, along with its ground truth and the model’s final answer, we employ OpenAI o4-mini to determine the correctness of the final answer with respect to the ground truth, with the prompt in Figure 8.

- C.3. Re-TRAC Details C.3.1. STRUCTURED STATE REPRESENTATION

As described in Section 4, the core of Re-TRAC is the trajectory compression mechanism that generates a structured state representation after each round. We design two variants of the state representation prompt tailored to different model capabilities.

Base Version (for Smaller and SFT Models) The base state representation (Figure 9) follows a fixed compression specification C that captures five complementary facets:

- • Current Answer: The best-supported partial answer identified so far, or “None” if no conclusive evidence exists.
- • Facts & Evidence Collected: All factual items discovered during the trajectory, with source annotations indicating provenance and verification status.
- • Analysis & Conclusions: Logical conclusions derived

- from the evidence, explicitly linked to supporting facts.
- • Source Inventory & Verification Status: An enumeration of all visited sources and their current verification status.
- • Uncertainties, Limitations, Gaps: Unknown variables, data ambiguities, and failure modes blocking a final decision.

Full Version with Audit Part (for Frontier LLMs) For frontier LLMs with stronger instruction-following and summarization capabilities, we extend the base version with three additional audit facets (Figure 10):

- • Failed Attempts: Specific plans or objectives that were abandoned, left unfinished, or resulted in no progress by the end of the rollout.
- • Uncompleted Proposals: Potential leads (URLs, entities, data points, or keywords) surfaced in tool outputs but never pursued due to token limits, focus shifts, or accidental omission.
- • Discarded Possibilities: Candidate answers or critical evidence that were discarded based on unverified assumptions, hallucinations, or logical leaps.

The state representation prompt is appended to the trajectory after the model reaches a stopping condition (either providing a final answer or hitting the context limit).

Table 8. Model-specific inference hyper-parameters for evaluation.

Model Context Window Temperature Top P Reasoning

- o4-mini 200k - - medium
- o3 200k - - medium GPT-5 400k - - medium DeepSeek-V3.2 140k 1.0 0.95 enabled GLM-4.7 128k 1.0 0.95 RE-TRAC-30B-A3B 128k 0.7 1.0 RE-TRAC-4B 128k 0.7 0.8 -

#### C.4. Evaluation Results

The test-time scaling methods (RT@N, MV@N, WV@N, Best@N) are defined in Section 5.2. In addition to these metrics and Pass@N, we introduce one additional metric specific to evaluating Re-TRAC in this appendix:

Accuracy Prefix (AP@N): Similar to Pass@N, AP@N measures whether at least one correct answer appears among the first N rounds. AP@N serves as an upper bound for ReTRAC performance and helps quantify how much room remains for improvement in answer selection strategies.

C.4.1. PER-MODEL DETAILED RESULTS

We present detailed round-by-round results for each model evaluated on BrowseComp300. Each table reports the accuracy at each individual round (Acc%), cumulative Pass@N, Re-TRAC accuracy (RT@N), Accuracy Prefix (AP@N), Majority Voting (MV@N), Weighted Voting (WV@N), and Best-of-N (Best@N) across 8 rounds.

Results for o4-mini, o3, GPT-5, DeepSeek-V3.2, and GLM4.7 are presented in Table 9, Table 10, Table 11, Table 12, and Table 13, respectively.

- C.3.2. CONTINUATION PROMPT

To enable cross-trajectory knowledge consolidation, we prepend the structured state representation from the previous round to the input of the subsequent round, along with a continuation prompt that guides the model’s utilization of this information. The continuation prompt template is shown in Figure 11.

- C.3.3. ROUND SETTINGS

In our experiments, we evaluate Re-TRAC with varying numbers of rounds. Unless otherwise specified, we set the maximum round limit K = 8 as the default setting for testtime scaling experiments. For SFT data construction, we collect trajectories with K = 4 from GLM-4.7.

- C.3.4. MODEL-SPECIFIC HYPER-PARAMETERS

We apply Re-TRAC to multiple frontier and open-source models. Table 8 summarizes the key hyper-parameters used for each model during inference.

Table 9. Round-by-round evaluation results for o4-mini on BrowseComp300.

Round / N Acc% Pass@N RT@N AP@N MV@N WV@N Best@N

- 1 26.7 26.7 26.7 26.7 26.7 26.7 26.7
- 2 26.1 36.0 33.8 34.1 26.7 32.7 32.7
- 3 25.3 41.3 38.8 39.1 29.7 36.3 36.0
- 4 24.0 46.3 42.5 42.8 29.0 40.7 39.3
- 5 25.5 48.3 42.5 43.1 29.7 42.0 40.0
- 6 25.1 50.7 44.8 45.5 29.0 43.0 40.3
- 7 26.3 54.7 45.2 46.5 29.0 42.3 42.0
- 8 26.4 57.3 46.8 47.8 30.3 44.7 43.7

Table 10. Round-by-round evaluation results for o3 on BrowseComp300.

Round / N Acc% Pass@N RT@N AP@N MV@N WV@N Best@N

- 1 56.7 56.7 56.7 56.7 56.7 56.7 56.7
- 2 54.7 67.3 61.1 62.1 56.7 64.7 64.7
- 3 51.3 72.0 65.4 66.1 58.7 65.3 64.7
- 4 54.7 75.0 67.1 67.4 60.0 66.7 64.0
- 5 56.0 77.7 67.1 68.1 62.0 68.7 66.7
- 6 54.9 79.3 68.1 69.1 62.0 68.7 66.7
- 7 56.3 80.7 70.1 71.1 63.7 70.0 68.0
- 8 54.7 81.7 69.8 71.1 64.3 70.0 68.0

Table 11. Round-by-round evaluation results for GPT-5 on BrowseComp300.

Round / N Acc% Pass@N RT@N AP@N MV@N WV@N Best@N

- 1 47.0 47.0 47.0 47.0 47.0 47.0 47.0
- 2 54.0 62.3 60.2 60.9 47.0 55.7 55.3
- 3 46.0 66.3 60.5 62.2 48.7 56.3 57.0
- 4 48.7 70.7 63.9 65.2 52.7 56.7 57.0
- 5 46.7 73.0 65.2 66.6 53.0 57.0 57.0
- 6 48.0 75.3 66.6 67.6 52.3 57.0 55.7
- 7 47.3 76.3 66.2 68.2 52.0 56.0 55.0
- 8 48.7 77.3 66.6 69.2 54.0 56.3 54.0

Table 12. Round-by-round evaluation results for DeepSeek-V3.2 on BrowseComp300.

Round / N Acc% Pass@N RT@N AP@N MV@N WV@N Best@N

- 1 44.3 44.3 44.3 44.3 44.3 44.3 44.3
- 2 46.0 56.7 52.8 54.5 44.3 52.7 52.7
- 3 44.7 62.7 56.2 59.0 46.7 58.3 57.0
- 4 44.0 65.7 58.7 60.4 48.7 60.0 59.3
- 5 47.7 68.3 58.0 61.1 47.7 59.3 60.3
- 6 42.7 69.7 59.4 62.2 48.3 58.0 59.7
- 7 45.0 70.7 60.1 63.2 49.0 56.7 59.7
- 8 47.7 72.0 60.8 63.9 50.7 58.0 60.3

Table 13. Round-by-round evaluation results for GLM-4.7 on BrowseComp300.

Round / N Acc% Pass@N RT@N AP@N MV@N WV@N Best@N

1 37.7 37.7 37.7 40.3 37.7 37.7 37.7 2 36.3 46.3 47.7 48.7 37.7 41.3 41.0 3 37.1 53.0 52.7 53.7 37.7 45.0 43.7 4 41.3 57.0 55.3 56.3 40.0 47.0 45.0 5 39.7 60.0 57.0 58.3 40.3 48.0 43.3 6 45.3 63.3 58.7 60.7 41.7 49.3 42.3 7 41.0 65.0 59.3 62.0 42.7 47.7 43.0 8 41.0 67.7 60.7 62.7 41.7 48.0 42.3

|I will show you a solution trajectory of another model that try to find the answer to a question via deep research. The model fails to give an correct answer finally. I need you to analyze the trajectory and detect whether the model has one of the following behaviors:<br><br>A. The model finally gives an answer or asks the user for more information. It has planned to try some other clues or possible branches during the process, but not all proposed clues and branches are explored in the end.<br>B. The model finally gives an answer or asks the user for more information. It has explored all clues and branches proposed during its search process.<br>C. The model finally hits the context limitation. It has planned to try some other clues or possible branches during the process, but finally stucks in an unproductive branch for a long time.<br>D. The model finally hits the context limitation. It has planned to try some other clues or possible branches during the process, and it is still exploring these different branches until hitting the context limitation.<br>E. None of the above.<br><br><br>Note that when the context limitation is hit, the user will warn the model and force the model to give an answer.<br><br>Your output should be in the following json format: {<br><br>"behavior": <A, B, C, D or E>, "reason": <A brief reason for your choice><br><br>} Here is the trajectory: {traj}|
|---|

Figure 6. Prompt for classifying trajectory failure modes in incomplete branch exploration analysis.

|Please process the following webpage content and user goal to extract relevant information:<br><br>## **Webpage Content**<br><br>{webpage content} ## **User Goal** {goal}<br><br>## **Task Guidelines**<br><br>1. **Content Scanning for Rationale**: Locate the specific sections/data directly related to the user’s goal within the webpage content<br>2. **Key Extraction for Evidence**: Identify and extract the most relevant information from the content, output the full original context as far as possible<br>3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity<br><br><br>**Final Output Format using JSON format has "rational", "evidence", "summary" fields**|
|---|

###### Figure 7. Prompt for the visit tool to extract and summarize goal-relevant information from webpage content.

|You are an evaluator. Based ONLY on the [correct answer], judge whether the [response] to the [question] is correct.<br><br>=== INPUTS === [question]: {question} [response]: {answer} [correct answer]: {ground truth}<br><br>=== TASK ===<br><br>1. Extract the single final answer from the [response]. If no clear final answer exists, write "None".<br>2. Give a concise explanation (reasoning) that ONLY compares the extracted answer with the [correct answer]. Do not solve the problem again or add extra background.<br><br>3. Decide correctness: set correctness = correct if they are equivalent / within a tiny numeric tolerance and acceptable difference of expression style; otherwise incorrect. [correct answer] may contain multiple answers separated by "OR", the response is correct if it matches any of the answers.<br><br>4. Extract a confidence score (0{100). If the [response] provides none, use 100.<br><br><br>=== OUTPUT FORMAT (STRICT) === Return a valid JSON object with exactly these keys: {<br><br>"extracted final answer": <string>, "reasoning": <string>, "correctness": <string "correct" or "incorrect">, "confidence": <integer 0-100><br><br>} Do NOT output anything else|no comments, no code fences.|
|---|

###### Figure 8. Prompt for answer verification using o4-mini, following the BrowseComp evaluation protocol.

|--- TASK COMPLETED / STARTING SUMMARIZE --Summary the trajectory above for the original question you are given: {input}. CRITICAL OUTPUT FORMAT REQUIREMENTS (STRICT) You MUST follow the exact format below. Do NOT add extra sections, headers, prefaces, or commentary. Do NOT use placeholders like "see above", "as mentioned", "ibid.", "refer to earlier", or "same as before". Do NOT omit any important information that appears in the input. OUTPUT FORMAT (EXACT) You must directly record the actual information content in plain text (not just summaries). Include:<br><br>0) Current Answer<br><br>- Provide the single best, most definite answer identified so far based on conclusive evidence. If no final answer is certain, state "None".<br><br>1) Facts & Evidence Collected<br><br>- List every factual item discovered. Attach a source annotation: [Source: <tool/doc> | <URL/id> | Verified: yes/no/partial].<br><br>2) Analysis & Conclusions<br><br>- State all logical conclusions derived, explicitly linking each to the supporting evidence IDs from section (1).<br><br>3) Source Inventory & Verification Status<br><br>- Enumerate all visited sources and evaluate their current verification status.<br><br>4) Uncertainties, Limitations, Gaps<br><br><br>- List all unknown variables, data ambiguities, and failure modes currently blocking a final decision. QUALITY RULES<br><br>- Be exhaustive over the input: no omissions.<br>- Keep claims tightly tied to evidence; if you infer, label it "Inference" and explain why.<br>- If sources conflict, present both and state which you trust more (and why).<br>|
|---|

###### Figure 9. Re-TRAC State Representation Prompt (base version without audit facets) used for smaller and SFT models.

|--- TASK COMPLETED / STARTING SUMMARIZE --Summary the trajectory above for the original question you are given: {input}. CRITICAL OUTPUT FORMAT REQUIREMENTS (STRICT) You MUST follow the exact format below. Do NOT add extra sections, headers, prefaces, or commentary. Do NOT use placeholders like "see above", "as mentioned", "ibid.", "refer to earlier", or "same as before". Do NOT omit any important information that appears in the input. OUTPUT FORMAT (EXACT) You must directly record the actual information content in plain text (not just summaries). Include:<br><br>0) Current Answer<br><br>- Provide the single best, most definite answer identified so far based on conclusive evidence. If no final answer is certain, state "None".<br><br>1) Facts & Evidence Collected<br><br>- List every factual item discovered. Attach a source annotation: [Source: <tool/doc> | <URL/id> | Verified: yes/no/partial].<br><br>2) Analysis & Conclusions<br><br>- State all logical conclusions derived, explicitly linking each to the supporting evidence IDs from section (1).<br><br>3) Source Inventory & Verification Status<br><br>- Enumerate all visited sources and evaluate their current verification status.<br><br>4) Uncertainties, Limitations, Gaps<br><br>- List all unknown variables, data ambiguities, and failure modes currently blocking a final decision.<br><br>5) Failed attempts<br><br>- Identify any specific plan or objective explicitly stated in the reasoning that was either abandoned, left unfinished, or resulted in no progress by the end of the rollout.<br><br>6) Uncompleted proposals<br><br>- Scan all tool outputs and internal reasoning for potential leads (URLs, entities, data points, or keywords) that were surfaced but never pursued due to token limits, focus on a different branch, or accidental omission.<br><br>7) Discarded possibilities<br><br><br>- Identify any candidate answers or critical evidence that were discarded based on unverified assumptions, hallucinations, or logical leaps. QUALITY RULES<br><br>- Be exhaustive over the input: no omissions.<br>- Keep claims tightly tied to evidence; if you infer, label it "Inference" and explain why.<br>- If sources conflict, present both and state which you trust more (and why).<br>|
|---|

###### Figure 10. Re-TRAC State Representation Prompt (full version with audit facets) used for frontier LLMs.

|Below is the summary from the previous attempt: {last summary}<br><br>1. Critical Evaluation:<br><br>The summary provided above is a *suggested* synthesis of past efforts, not an absolute truth. It may contain hallucinations, unverified assumptions, or premature conclusions. You are encouraged to:<br><br>- Identify and ignore any parts of the summary that seem illogical or poorly supported.<br>- If the summary’s quality is low or its direction feels like a dead-end, you have the full autonomy to completely disregard it and initiate a fresh search strategy.<br><br><br>2. Expand the search space:<br><br>If the task remains unsolved, it means the current search space is insufficient. Do not get trapped in the "logic loop" of the summary. Use the [Uncertainties, Limitations, Gaps] section as a springboard to:<br><br>- Pivot to entirely different keyword clusters or tool-call strategies.<br>- Cross-verify "Facts" that were marked as "unverified" or "partial" in the<br><br><br>summary.<br><br>3. Autonomous re-planning:<br><br><br>The summary provides the *memory*, but you provide the *reasoning*. Based on the existing facts and your own judgment of the current situation, determine your own next steps.<br><br>Now continue your deep search task. You have full permission to be skeptical of the past and bold in your current exploration.|
|---|

###### Figure 11. Re-TRAC Continuation Prompt prepended to subsequent rounds to guide cross-trajectory knowledge consolidation.

