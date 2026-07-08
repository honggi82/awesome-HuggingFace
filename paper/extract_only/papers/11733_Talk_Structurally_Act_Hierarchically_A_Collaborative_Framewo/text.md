## Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems

Zhao Wang∗,†, Sota Moriyama∗, Wei-Yao Wang, Briti Gangopadhyay, Shingo Takamatsu Sony Group Corporation, Japan

# arXiv:2502.11098v1[cs.AI]16Feb2025

### Abstract

Recent advancements in LLM-based multiagent (LLM-MA) systems have shown promise, yet significant challenges remain in managing communication and refinement when agents collaborate on complex tasks. In this paper, we propose Talk Structurally, Act Hierarchically (TalkHier), a novel framework that introduces a structured communication protocol for context-rich exchanges and a hierarchical refinement system to address issues such as incorrect outputs, falsehoods, and biases. TalkHier surpasses various types of SoTA, including inference scaling model (OpenAI-o1), open-source multi-agent models (e.g., AgentVerse), and majority voting strategies on current LLM and single-agent baselines (e.g., ReAct, GPT4o), across diverse tasks, including open-domain question answering, domainspecific selective questioning, and practical advertisement text generation. These results highlight its potential to set a new standard for LLM-MA systems, paving the way for more effective, adaptable, and collaborative multiagent frameworks. The code is available at https://github.com/sony/talkhier.

### 1 Introduction

Large Language Model (LLM) Agents have broad applications across domains such as robotics (Brohan et al., 2022), finance (Shah et al., 2023; Zhang et al., 2024b), and coding (Chen et al., 2021; Hong et al., 2023). By enhancing capabilities such as autonomous reasoning (Wang et al., 2024b) and decision-making (Eigner and Händler, 2024), LLM agents bridge the gap between human intent and machine execution, generating contextually relevant responses (Pezeshkpour et al., 2024).

Recent research has primarily focused on LLMbased Multi-Agent (LLM-MA) systems, which

*These authors contributed equally to this work †Corresponding author: Zhao Wang (Email Address:

Zhao.Wang@sony.com)

###### Structured Protocol:

###### Problem 1: Disorganized, Long Text

- 1) Background: …
- 2) Output format: …
- 3) Message: ….

based Communication Protocol.

| |
|---|

[Figure 1]

| |
|---|

| |
|---|

[Figure 2]

[Figure 3]

Problem 2: Sequential or Similar

Problem 2: Problem 2: Agents Share Same Memory:

[Figure 4]

Evaluators Work

[Figure 5]

| |
|---|

Predefined Evaluators

| |
|---|

Hierarchically,

- Agent 1 did…at Step 1
- Agent 2… …

Independent Memory 1

Independent Memory 1

###### Fail to Manage or

Ensuring Controlled, Well-Summarized

[Figure 6]

Balance Opinions.

| |
|---|

[Figure 7]

Opinion Flow.

| |
|---|

| |
|---|

- Figure 1: Existing LLM-MA methods (left) face two major challenges: 1) disorganized, lengthy text-based communication protocols, and 2) sequential or overly similar flat multi-agent refinements. In contrast, TalkHier (right) introduces a well-structured communication protocol and a hierarchical refinement approach.

|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>88.38<br><br>83.66 83.6<br><br>71.98<br><br>65.17<br><br>70.07 70.44 70.71 71.15<br><br>58.5 65.91 66.82 67.19<br><br>Accuracy(%)on MMLU<br><br>+ 5.64%<br><br>Open AI o1: 87.56|
|---|

- Figure 2: Our TalkHier built on GPT4o surpasses inference scaling models (OpenAI-o1), open-source multiagent models (AgentVerse and etc.), and models with majority voting strategies (ReAct, GPT4o) on five subtasks of MMLU.

leverage collective intelligence and specialize each agent with the corresponding subtasks, to solve complicated and multi-step problems. For instance, previous works on LLM-MA have explored approaches where instances of LLMs, referred to as agents (Xi et al., 2023; Gao et al., 2023; Wang et al., 2024a; Cheng et al., 2024; Ma et al., 2024), collaborate synergistically by debate (Chen et al., 2024), reflection (He et al., 2024), self-refinement (Madaan et al., 2023), or multi-agent based feedback refinement (Yang et al., 2023). These systems employ diverse communication topologies to enable efficient interactions between agents such as

Chain (Wei et al., 2022) and Tree (Yao et al., 2023) structures, among others (Qian et al., 2024; Zhuge et al., 2024; Zhang et al., 2024a).

Despite the promising advancements in LLMMA systems, several challenges in this field remain unexplored (shown in Figure 1):

- 1) Disorganized communication in text form. Agents often engage in debates (Zhao et al., 2024), share insights (Chen et al., 2024), or perform refinement (Madaan et al., 2023; Yang et al., 2023) to effectively solve complex tasks, with their exchanges primarily in text form (Guo et al., 2024). However, communication often becomes disorganized because it requires explicitly describing agent tasks, providing background context for the communication, and specifying the required output formats. These factors together lead to lengthy and unstructured exchanges, making it difficult for agents to manage subgoals, maintain output structures, and retrieve independent memories from prior actions and observations.
- 2) Refinement schemes. While some studies have shown that incorporating agent debates (Chen et al., 2024) or evaluation-based multi-agent refinement (Wang et al., 2023; Yang et al., 2023) can improve system accuracy, these approaches also expose significant limitations. As the number of agents increases, LLM-MA systems face challenges in effectively summarizing opinions or feedback (Fang et al., 2024). They often fail to balance these inputs, frequently overlooking some or exhibiting biases based on the order in which feedback is provided (Errica et al., 2024).

In this paper, we propose a novel collaborative LLM-MA framework called Talk Structurally, Act Hierarchically (TalkHier)-the first collaborative LLM-MA framework to integrate a well-structured communication protocol with hierarchical refinement. Our key contributions shown in Figure 1 and

- 2 are as follows:

1. Well-Structured, Context-Rich Communication Protocol: TalkHier introduces a novel communication protocol that incorporates newly proposed elements: messages, intermediate outputs, and relevant background information. These components form the foundation of a wellstructured protocol that organizes agent communication, ensuring clarity and precision. By embedding these elements, TalkHier significantly improves communication accuracy and efficiency compared to traditional text-based

methods.

- 2. Hierarchical Refinement in LLM-MA Systems: TalkHier enhances traditional multi-agent evaluation systems with a hierarchical refinement framework, enabling agents to act hierarchically. This approach addresses such as the difficulty in summarizing opinions or feedback as the number of agents increases, balancing diverse inputs, and mitigating biases caused by the order of feedback processing, resulting in more reliable and robust interactions.
- 3. State-of-the-Art Results Across Benchmarks: Experimental results show that TalkHier achieves state-of-the-art performance on diverse benchmarks, including selective problemsolving in complex sub-domains, open question answering, and Japanese text generation tasks. Ablation studies confirm the effectiveness of each component, demonstrating their contributions to the framework’s overall success.

### 2 Related Work

Collaborative LLM-MA. LLM-MA systems enable agents to collaborate on complex tasks through dynamic role allocation, communication, and task execution (Guo et al., 2024; Han et al., 2024). Recent advancements include agent profiling (Yang et al., 2024), hierarchical communication (Rasal, 2024), and integration of reasoning and intentions (Qiu et al., 2024). However, challenges remain in ensuring robust communication, avoiding redundancy, and refining evaluation processes (Talebirad and Nadiri, 2023). Standardized benchmarks and frameworks are needed to drive future progress (Li et al., 2024).

Communication in LLM-MA. Effective communication is crucial for collaborative intelligence (Guo et al., 2024). While many previous works, including chain (Wei et al., 2022), tree (Yao et al., 2023), complete graph (Qian et al., 2024), random graph (Qian et al., 2024), optimizable graph (Zhuge et al., 2024), and pruned graph (Zhang et al., 2024a) methods have focused on communication topologies, there has been limited discussion on the optimal form of communication. Most systems rely on text-based exchanges (Zhang et al., 2024a; Shen et al., 2024), which is inefficient and prone to errors as agents often lose track of subtasks or fail to recall prior outputs as tasks grow in complexity. We argue

|Prompt (General Task Assignment)|
|---|

|Prompt (General Task Assignment)| |
|---|---|
| |①”Your sub task is…You<br><br>(Only-text based, Length Task<br><br>② “The output of mine is :…”<br><br>Bad Response due to<br><br>|

How Many?

To finish this I need ….

In whatFormat? …

① Well-organized communication protocol

need….”

?

- a. (Message With Specified Subtask): “Your Subtask is to …the format is….”
- b. (Intermediate Output) : “…”
- c. (Background): “We are discussing …”

)

[Figure 47]

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

Member Agent 1 (Role: Generator)

Member Agent 1

Non-organized

instruction

- ②Accurate Response with Intermediate Output

- ③ Well-organized communication protocol

| | | |
|---|---|---|
| | | |

- ③ ”Your sub task is…You need….” (Only-text based, Length Task)

Member Agent 2

(Role: Evaluator)

Metircs?

Evaluate what? …

- ④ “The output of mine is :…” Bad Response due to Non-organized instruction

?

These are good as…

| | | |
|---|---|---|
| | | |

- a. (Message With Specified Subtask): “Your Subtask is to evaluate …the format…”
- b. (Intermediate Output): “…”
- c. (Background):“We are discussing …” Member Agent 2

| | | |
|---|---|---|
| | | |

Supervisor

Supervisor

[Figure 48]

…

| | | |
|---|---|---|
| | | |

④ Accurate Response with Intermediate Output

###### Context-Poor Communication:① Task Assignment: Assigning the

next agent without providing well-organized information. ② Bad

Structured Communication: ① Task assignment: Assigning specific subtask

Responses: Producing poor responses due to forgetting key issues,

###### to the next agent with message, intermediate output and background.

such as the context of the question or intermediate format.

② Accurate responses: including response and other intermediate output.

- ① “Generate text….”

- ② “ My generated text is….”

Member Agent 1

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

(Role: Generator)

① Task: “Generate text….”

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

Supervisor

- ④-a “ Evaluate it from

- {placeholder= Criterion 1}

⑤-b“ The Score of

- {placeholder= Criterion 2} is …..”

“Evaluate thetext…….③”

}isgood,but the{placeholder=1

}isnot good” Criterion2

{placeholder=CriterionThe⑥“

Supervisor Agent 1 (Role: Generator)

| | | |
|---|---|---|
| | | |

② “ My Generated text is….” ⑤-a“ The Score of

④ “ My evaluated result is….”

- Member Agent 2

- (Role: Evaluator 1)

Member Agent 3

- (Role: Evaluator 2)

{placeholder= Criterion 1} is …..”

③ “ My evaluated result is….”

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

④-b “ Evaluate it from {placeholder= Criterion 2}

| | | |
|---|---|---|
| | | |

Agent 2 (Role: Evaluator 1)

Agent 3 (Role: Evaluator 2 )

| | | |
|---|---|---|
| | | |

Evaluation Team

###### Traditional Evaluation based Refinement: All agents follow a

…

Supervisor

predefined order, such as Supervisor → Evaluator 1 → Evaluator 2, or

similar styles, without summarization or coordination. This leads to

Hierarchy Refinement: An evaluator team operates hierarchically, providing

biased outcomes (sensitivity to evaluator order) or incorrect results

summarized and coordinated feedback, balancing diverse inputs, and

(overlooking partial evaluations).

mitigating biases, which collectively enhance overall task quality.

Figure 3: Comparisons between existing approaches (left) and ours (right). Our TalkHier proposes a new communication protocol (first row) featuring context-rich and well-structured communication information, along with a collaborative hierarchical refinement (second row) where evaluations provide summarized and coordinated feedback within an LLM-MA framework.

c(ijt) ∈ Cp represents a communication between agents vi and vj along an edge eij ∈ E at time step t. While the graph structure G remains fixed, the communication events Cp are dynamic and adapt to the specific task.

for structured communication protocols that guide subtasks with clear, context-specific instructions, ensuring coherence across interactions.

Feedback-Based Refinement. Feedback mechanisms, such as Self-Refine (Madaan et al., 2023) and generator-evaluator frameworks (Wang et al., 2023), improve system accuracy through iterative refinement. However, these methods face challenges in managing diverse feedback, which can lead to bias or inefficiencies if inputs are not wellorganized (Xu et al., 2024). Scalable, unbiased solutions are essential to enhance multi-agent evaluation processes.

#### 3.1 Agents with Independent Memory

Each agent vi ∈ V in graph G can be formally represented as:

vi = (Rolei,Pluginsi,Memoryi,Typei).

Rolei: Assign roles such as generator, evaluator, or revisor based on the task type. Pluginsi: External tools or plugins attached for domain-specific operations. Memoryi: An agent-specific memory that stores and retrieves information relevant to the agent’s role and task. Typei: Specifies whether the agent is a Supervisor (S) responsible for overseeing task success, or a Member (M) focused on problem-solving.

### 3 Methodology

TalkHier aims to design a LLM-MA system represented as a graph G = (V,E), where V denotes the set of agents (nodes) and E represents the set of communication pathways (edges). Given an input problem p, the system dynamically defines a set of communication events Cp, where each event

The first two components—Rolei, and Pluginsi—are standard in most related works,

###### Supervisor Prompt Template

Team Members: [Description of each team member’s role] Conversation History: [Independent Conversation History] Given the conversation above, output the following in this exact order:

- 1. ‘thoughts’: Output a detailed analysis on the most recent message. In detail, state what you think should be done next, and who you think you should contact next.
- 2. Who should act next? Select one of: [Team member names] and output as ‘next’. When you have determined that the final output is gained, report back with FINISH.
- 3. ‘messages’: If the next agent is one of [Team member names], give detailed instructions. If FINISH, report a summary of all results.
- 4. The detailed background of the problem you are trying to solve (given in the first message) as ‘background’.
- 5. The intermediate outputs to give as ‘intermediate_output’.

Member Prompt Template

[Role of member] Background: [Background information given by Supervisor] Conversation History: [Independent Conversation History]

Figure 4: Prompts for acquiring the contents of the context-rich, structured communication protocol in TalkHier.

forming the foundation of agent functionality. Our contributions lie in the last three components: Memoryi, which equips each agent with our refined independent, agent-specific memory for reasoning, Teami, which represents the team the agent is a part of, and Typei, which explicitly categorizes agents into Supervisor (S) roles, responsible for overseeing the multi-agent team and ensuring task success, or Member (M) roles, focused on problem-solving and optionally utilizing plugins. These additions enable hierarchical, structured collaboration and role-specific operations within the framework.

Agent-Specific Memory. To enhance efficiency and scalability, each agent vi maintains an independent memory, Memoryi. Unlike long-term memory, which relies on a shared memory pool accessible by all agents, or short-term memory, which is limited to a single session or conversational thread, our proposed memory mechanism is agent-specific but not limited to session or conversational thread.

TalkHier allows each agent to independently retain and reason on its past interactions and knowledge, offering two key advantages: independence, where each agent’s memory operates without interference from others, avoiding centralized dependencies; and persistence, enabling agents to maintain historical data across sessions for consistent and informed decision-making.

3.2 Context-Rich Communication Between Agents

Communication between agents is represented by communication events c(ijt) ∈ Cp, where each event c(ijt) encapsulates the interaction from agent vi to agent vj along an edge eij ∈ E at time step t. Formally, a communication event c(ijt) is defined as:

c(ijt) = (M(ijt),B(ijt),I(ijt)),

where M(ijt) indicates the message content sent from vi to vj, containing instructions or clarifica-

tions, B(ijt) denotes background information to ensure coherence and task progression, including the

problem’s core details and intermediate decisions, and I(ijt) refers to the intermediate output generated by vi, shared with vj to support task progression and traceability, all at time step t. These structures ensure that agents of TalkHier accomplish efficient communication and task coordination.

Communication Event Sequence. At each time step t, the current agent vi communicates with a connected node vj, with one being selected by the LLM if more than one exists. The elements of each edge M(ijt),B(ijt) and I(ijt) are then generated by invoking an independent LLM. To ensure consistency, clarity, and efficiency in extracting these elements, the system employs specialized prompts tailored to the roles of Supervisors and Members, as illustrated in Figure 4. Most notably, background information B(ijt) is not present for connections

Algorithm 1: Hierarchical Refinement

Input: Initial output A0 generated by the Generator node vmainGen , quality threshold Mthreshold, maximum iterations Tmax Output: Final output Afinal

- 1 Initialize iteration counter t ← 0
- 2 repeat

- 3 t ← t + 1 // Step 1: Task Assignment from vmains to vevals
- 4 T(assignt) = {(RolevS

eval

, CriteriavS

eval

)} // Step 2: Task Distribution by vevals

- 5 T(distributet) = {(CriterionvEi

eval

)}ki=1 // Step 3: Evaluation

- 6 F(t)

vevalEi

= fevaluate(At−1, CriterionvEi

eval

), ∀vevalEi ∈ Veval

- 7 F(evalt) = {F(t) vevalE1

, . . . , F(t)

vevalEk

} // Step 4: Feedback Aggregation by vevals

- 8 Fevalsummary = fsummarize(F(evalt) ) // Step 5: Summarizing results
- 9 if M(Fevalsummary) ≥ Mthreshold then

- 10 return Afinal = At−1 // Step 6: Return the current text if above threshold

- 11 At = frevise(At−1, Fevalsummary) // Step 7: Revision of the text
- 12 until t ≥ Tmax
- 13 return Afinal = At

[Figure 49]

Figure 5: Illustrated hierarchy of TalkHier.

from Member nodes to Supervisor nodes. These information are then established as a communication event c(ijt) ∈ Cp.

- 3.3 Collaborative Hierarchy Agent Team The entire graph G consists of multiple teams, each represented as a subset Vteam ⊆ V. Each team includes a dedicated supervisor agent vteamS and one

or more member agents vteamM . A key feature of the hierarchical structure in TalkHier is that a member agent in one team can also act as a supervisor for another team, creating a nested hierarchy of agent teams. As shown in the second row of Figure 3, this structure enables the entire graph G to represent a hierarchical node system, where teams are recursively linked through supervisor-member relationships.

Formally, the hierarchical structure of agents with two teams is defined as:

Vmain = {vmainS ,vmainGen ,vevalS ,vmainRev }, Veval = {vevalS ,vevalE1 ,vevalE2 ,...,vevalEk },

where the Main Supervisor (vmainS ) and Evaluation Supervisor (vevalS ) oversee their respective team’s

operations and assign tasks to each member, the Generator (vmainGen ) gives solutions for a given problem, and the Revisor (vmainRev ) refines outputs based on given feedback. Furthermore, the evaluation team is composed of k independent evaluators vevalEk , each of which outputs evaluation results for a given problem based on their specified metric. The overall structure is shown in Figure 5.

Algorithm. Algorithm 1 illustrates the operation of our hierarchical refinement process within the collaborative agent framework. The process begins with the main Supervisor (vmainS ) assigning tasks to the evaluation Supervisor (vevalS ), who then distributes evaluation criteria to individual evaluators (vevalEi ). Each evaluator assesses the generated output (At−1) based on their assigned criteria, producing detailed feedback. The evaluation Supervisor aggregates and summarizes this feedback (Fevalsummary) before passing it to the main Supervisor. The main Supervisor evaluates whether the summarized feedback meets the quality threshold (Mthreshold). If the threshold is satisfied, the output is finalized; otherwise, the Revisor (vmainRev ) refines the output for further iterations. This iterative refinement ensures accurate and unbiased collaboration across the agent hierarchy.

The main Supervisor evaluates whether the summarized feedback meets the quality threshold (Mthreshold), defined vaguely as “ensuring correctness” or “achieving high relevance.” If satisfied, the output is finalized; otherwise, the Revisor (vmainRev ) refines it. Details of our settings are in Appendix B, Appendix C, and Appendix D.

### 4 Experiments

In this section, we aim to answer the following research questions across various domains:

- RQ1: Does TalkHier outperform existing multiagent, single-agent, and proprietary approaches on general benchmarks?
- RQ2: How does TalkHier perform on open-domain question-answering tasks?
- RQ3: What is the contribution of each component of TalkHier to its overall performance?
- RQ4: How well does TalkHier generalize to more practical but complex generation task?

#### 4.1 Experimental Setup

Datasets. We evaluated TalkHier on a diverse collection of datasets to assess its performance across various tasks. The Massive Multitask Language Understanding (MMLU) Benchmark (Hendrycks et al., 2021) tests domain-specific reasoning problems including Moral Scenario, College Physics, Machine Learning, Formal Logic and US Foreign Policy. WikiQA (Yang et al., 2017) evaluates opendomain question-answering using real-world questions from Wikipedia. The Camera Dataset (Mita

- et al., 2024) focuses on advertisement headline generation, assessing the ability to create high-quality advertising text.

Baselines. To evaluate TalkHier, we compared it against a comprehensive set of baselines including:

- • GPT-4o (OpenAI, 2024a), based on OpenAI’s GPT-4 model with both single-run and ensemble majority voting (3, 5, or 7 runs).
- • OpenAI-o1-preview (OpenAI, 2024b), a beta model using advanced inference techniques, though limited by API support.
- • ReAct (Yao et al., 2022), a reasoning and action framework in single-run and ensemble configurations.
- • AutoGPT (Gravitas, 2023), an autonomous agent designed for task execution and iterative improvement.
- • AgentVerse (OpenBMB, 2023), a multi-agent system framework for collaborative problemsolving.
- • GPTSwarm (Zhuge et al., 2024), a swarmbased agent collaboration model utilizing optimizable communication graphs.
- • AgentPrune (Zhang et al., 2024a), a model leveraging pruning techniques for efficient multi-agent communication and reasoning.

Table 1: General Performance on MMLU Dataset. The table reports accuracy (%) for various baselines across Moral Scenario (Moral), College Physics (Phys.), Machine Learning (ML), Formal Logic (FL) and US Foreign Policy (UFP) domains. The notations 3@, 5@, and 7@ represent majority voting results using 3, 5, and 7 independent runs, respectively.

Models Moral Phys. ML FL UFP Avg. GPT4o 64.25 62.75 67.86 63.49 92.00 70.07 GPT4o-3@ 65.70 62.75 66.07 66.67 91.00 70.44 GPT4o-5@ 66.15 61.76 66.96 66.67 92.00 70.71 GPT4o-7@ 65.81 63.73 66.96 68.25 91.00 71.15 ReAct 69.61 72.55 59.82 32.54 58.00 58.50 ReAct-3@ 74.75 83.33 66.07 52.38 53.00 65.91 ReAct-5@ 74.97 82.35 66.96 46.83 63.00 66.82 ReAct-7@ 75.53 84.78 67.86 50.79 57.00 67.19 AutoGPT 66.37 78.43 64.29 60.83 90.00 71.98 AgentVerse 79.11 93.14 79.46 78.57 88.00 83.66 GPTSwarm 60.48 67.70 72.32 68.33 57.00 65.17 AgentPrune 70.84 91.18 81.25 81.75 93.00 83.60 o1-preview 82.57 91.17 85.71 83.33 95.00 87.56 TalkHier (Ours) 83.80 93.14 84.68 87.30 93.00 88.38

• OKG (Wang et al., 2025), A method tailored specifically for ad text generation tasks and easily generalizable to ad headlines with minimal prompt redefinition.

Implementation details. For fair comparisons, we use GPT-4o as the backbone across all experiments for the baselines and TalkHier, with the temperature set to 0 in all settings. For the OpenAI-

- o1 baseline, we followed the implementation guide and the limitations outlined in OpenAI’s documentation†, and keep the temperature fixed at 1.

4.2 Performance on MMLU (RQ1)

Table 1 reports the average accuracy of various models on the five domains of MMLU dataset. TalkHier, built on GPT-4o, achieves the highest average accuracy (88.38%), outperforming

- open-source multi-agent models (e.g., AgentVerse, 83.66%) and majority voting strategies applied to current LLM and single-agent baselines (e.g., ReAct-7@, 67.19%; GPT-4o-7@, 71.15%). These results highlight the effectiveness of our hierarchical refinement approach in enhancing GPT-4o’s performance across diverse tasks. Although OpenAI-
- o1 cannot be directly compared to TalkHier and
- other baselines—since they are all built on GPT-4o and OpenAI-o1’s internal design and training data remain undisclosed—TalkHier achieves a slightly higher average score (88.38% vs. 87.56%), demon-

†https://platform.openai.com/docs/guides/ reasoning/beta-limitations

Table 2: Evaluation Results on WikiQA. The table reports Rouge-1 and BERTScore for various models.

Models Rouge-1 BERTScore

GPT4o 0.2777 0.5856 ReAct 0.2409 0.5415 AutoGPT 0.3286 0.5885 AgentVerse 0.2799 0.5716 AgentPrune 0.3027 0.5788 GPTSwarm 0.2302 0.5067 o1-preview 0.2631 0.5701 TalkHier (Ours) 0.3461 0.6079

strating competitive performance.

#### 4.3 Evaluation on WikiQA Benchmark (RQ2)

We evaluated TalkHier and baselines on the WikiQA dataset, an open-domain question-answering benchmark. Unlike MMLU, WikiQA requires generating textual answers to real-world questions. The quality of generated answers was assessed using two metrics: Rouge-1 (Lin, 2004), which measures unigram overlap between generated and reference answers, and BERTScore (Zhang et al., 2020), which evaluates the semantic similarity between the two.

Table 2 shows that TalkHier outperforms baselines in both Rouge-1 and BERTScore, demonstrating its ability to generate accurate and semantically relevant answers. While other methods, such as AutoGPT and AgentVerse, perform competitively, their scores fall short of TalkHier, highlighting its effectiveness in addressing open-domain questionanswering tasks.

#### 4.4 Ablation Study (RQ3)

To better understand the contribution of individual components in TalkHier, we conducted ablation studies by removing specific modules and evaluating the resulting performance across the Moral Scenario, College Physics, and Machine Learning domains. The results of these experiments are summarized in Table 3.

Table 3 presents the contributions of our ablation study on the main components in TalkHier. Removing the evaluation Supervisor (TalkHier w/o Eval. Sup.) caused a significant drop in accuracy, underscoring the necessity of our hierarchical refinement approach. Replacing the structured communication protocol with the text-based protocol (TalkHier w. Norm. Comm) resulted in moderate accuracy reductions, while eliminating the entire evaluation

- Table 3: Ablative Results on Main Components of TalkHier: Accuracy (%) across Physics, ML, and Moral domains. TalkHier w/o Eval. Sup. removes the evaluation supervisor. TalkHier w/o Eval. Team excludes the evaluation team component. TalkHier w. Norm. Comm uses a normalized communication protocol.

Models Moral Phys. ML Avg. w/o Eval. Sup. 83.57 87.25 74.77 81.86 w/o Eval. Team 73.54 80.34 74.56 76.15 w. Norm. Comm 82.91 88.24 82.14 84.43 React (Single Agent) 69.61 72.55 59.82 67.33 TalkHier (Ours) 83.80 93.14 84.68 87.21

- Table 4: Ablative Results: Accuracy (%) across Physics, ML, and Moral domains. The study examines the impact of removing components from the structured communication protocol: message (Mij), background (Bij), and intermediate output (Iij).

Models Moral Phys. ML Avg. w/o Iij 81.56 90.20 75.89 82.55 w/o Bij 76.87 87.50 70.54 78.30 w/o Bij, Iij 77.99 90.20 78.57 82.25 TalkHier (Ours) 83.80 93.14 84.68 87.21

team (TalkHier w/o Eval.Team) led to substantial performance declines across all domains. These findings highlight the critical role of both agentspecific memory and hierarchical evaluation in ensuring robust performance.

Table 4 delves into the impact of individual elements in the communication protocol. Removing intermediate outputs (TalkHier w/o Iij) or background information (TalkHier w/o Bij) lead to inferior performance, with their combined removal (TalkHier w/o Bij,Iij) yielding similar declines. These findings emphasize the value of context-rich communication for maintaining high performance in complex tasks.

4.5 Evaluation on Ad Text Generation (RQ4) We evaluate TalkHier on the Camera dataset (Mita et al., 2024) using traditional text generation metrics (BLEU-4, ROUGE-1, BERTScore) and domain-specific metrics (Faithfulness, Fluency, Attractiveness, and Character Count Violation) (Mita et al., 2024). These metrics assess both linguistic quality and domain-specific relevance.

Setting up baselines like AutoGPT, AgentVerse, and GPTSwarm for this task was challenging, as their implementations focus on general benchmarks like MMLU and require significant customization

Table 5: Evaluation Results on Camera Dataset. We report BLEU-4 (B4), ROUGE-1 (R1), BERTScore (BERT), and domain-specific metrics (Faithfulness, Fluency, Attractiveness, Character Count Violation(CCV)) following (Mita et al., 2024).

Models B4 (↑) R1 (↑) BERT (↑) Faithfulness (↑) Fluency (↑) Attractiveness (↑) CCV (↓)

GPT-4o 0.01 0.02 0.65 4.8 5.9 6.5 16% ReAct 0.01 0.01 0.70 4.9 6.4 7.0 17% OKG 0.03 0.16 0.73 6.3 8.7 6.1 4% TalkHier (Ours) 0.04 0.20 0.91 8.6 8.9 6.2 4%

for ad text generation. In contrast, OKG (Wang

- et al., 2025), originally for ad keyword generation, was easier to adapt, making it a more practical baseline.

Table 5 presents the results. TalkHier outperforms ReAct, GPT-4o, and OKG across most metrics, particularly excelling in Faithfulness, Fluency, and Attractiveness while maintaining a low Character Count Violation rate. The mean performance gain over the best-performing baseline, OKG, across all metrics is approximately 17.63%.

To verify whether TalkHier’s multi-agent evaluations of attractiveness, fluency, and faithfulness are accurate, we conducted a subjective experiment on a sub-dataset of Camera, comparing the system’s automatic ratings to human judgments; details of this procedure are provided in Appendix E.

### 5 Discussion

The experimental results across the MMLU, WikiQA, and Camera datasets consistently demonstrate the superiority of TalkHier. Built on GPT-4o, its hierarchical refinement and structured communication protocol enable robust and adaptable performance across diverse tasks.

General and Practical Benchmarks. TalkHier outperformed baselines across general and practical benchmarks. On MMLU, it achieved the highest accuracy (88.38%), surpassing the best opensource multi-agent baseline, AgentVerse (83.66%), by 5.64%. On WikiQA, it obtained a ROUGE-1 score of 0.3461 (+5.32%) and a BERTScore of 0.6079 (+3.30%), outperforming the best baseline, AutoGPT (0.3286 ROUGE-1, 0.5885 BERTScore). On the Camera dataset, TalkHier exceeded OKG across almost all metrics, demonstrating superior Faithfulness, Fluency, and Attractiveness while maintaining minimal Character Count Violations. These results validate its adaptability and taskspecific strengths, highlighting its advantage over inference scaling models (e.g., OpenAI-o1), open-

source multi-agent models (e.g., AgentVerse), and majority voting strategies (e.g., ReAct, GPT-4o).

Comparative and Ablation Insights. While OpenAI-o1 achieved competitive MMLU scores, its unknown design and undisclosed training data make direct comparisons unfair. Since TalkHier is built on the GPT-4o backbone, comparisons with other GPT-4o-based baselines are fair. Despite this, TalkHier was competitive with OpenAI-o1 on MMLU and achieved a significant advantage on WikiQA. Ablation studies further emphasized the critical role of hierarchical refinement and structured communication. Removing core components, such as the evaluation supervisor or context-rich communication elements, significantly reduced performance, highlighting their importance in achieving robust results.

### 6 Conclusions

In this paper, we propose TalkHier, a novel framework for LLM-MA systems that addresses key challenges in communication and refinement. To the best of our knowledge, TalkHier is the first framework to integrate a structured communication protocol in LLM-MA systems, embedding Messages, intermediate outputs, and background information to ensure organized and context-rich exchanges. At the same time, distinct from existing works that have biases on inputs, its hierarchical refinement approach balances and summarizes diverse opinions or feedback from agents. TalkHier sets a new standard for managing complex multi-agent interactions across multiple benchmarks, surpassing the best-performing baseline by an average of 5.64% on MMLU, 4.31% on WikiQA, and 17.63% on Camera benchmarks. Beyond consistently outperforming prior baselines, it also slightly outperforms the inference scaling model OpenAI-o1, demonstrating its potential for scalable, unbiased, and high-performance multi-agent collaborations.

### Limitations

One of the main limitations of TalkHier is the relatively high API cost associated with the experiments (see Appendix A for details). This is a tradeoff due to the design of TalkHier, where multiple agents collaborate hierarchically using a specifically designed communication protocol. While this structured interaction enhances reasoning and coordination, it also increases computational expenses.

This raises broader concerns about the accessibility and democratization of LLM research, as such costs may pose barriers for researchers with limited resources. Future work could explore more costefficient generation strategies while preserving the benefits of multi-agent collaboration.

### References

Anthony Brohan et al. 2022. Code as policies: Language model-driven robotics. arXiv preprint arXiv:2209.07753.

Mark Chen et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Pei Chen, Boran Han, and Shuai Zhang. 2024. Comm: Collaborative multi-agent, multi-reasoningpath prompting for complex problem solving. arXiv preprint arXiv:2404.17729.

Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui Meng, Sirui Hong, Wenhao Li, Zihao Wang, Zekai Wang, Feng Yin, Junhua Zhao, and Xiuqiang He. 2024. Exploring large language model based intelligent agents: Definitions, methods, and prospects. CoRR, abs/2401.03428.

Eva Eigner and Thorsten Händler. 2024. Determinants of llm-assisted decision-making. arXiv preprint arXiv:2402.17385.

Federico Errica, Giuseppe Siracusano, Davide Sanvito, and Roberto Bifulco. 2024. What did i do wrong? quantifying llms’ sensitivity and consistency to prompt engineering. arXiv preprint arXiv:2406.12334.

Jiangnan Fang, Cheng-Tse Liu, Jieun Kim, Yash Bhedaru, Ethan Liu, Nikhil Singh, Nedim Lipka, Puneet Mathur, Nesreen K Ahmed, Franck Dernoncourt, et al. 2024. Multi-llm text summarization. arXiv preprint arXiv:2412.15487.

Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li. 2023. Large language models empowered agent-based modeling and simulation: A survey and perspectives. CoRR, abs/2312.11970.

Significant Gravitas. 2023. Autogpt: An experimental open-source application.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680.

Shiyang Han, Qian Zhang, Yue Yao, Wenhao Jin, Zhen Xu, and Cheng He. 2024. Llm multi-agent systems: Challenges and open problems. arXiv preprint arXiv:2402.03578.

Chengbo He, Bochao Zou, Xin Li, Jiansheng Chen, Junliang Xing, and Huimin Ma. 2024. Enhancing llm reasoning with multi-path collaborative reactive and reflection agents. arXiv preprint arXiv:2501.00430.

Dan Hendrycks, Colin Burns, Samuel Basart, Chia Zou, David Song, and Thomas G. Dietterich. 2021. Measuring massive multitask language understanding. arXiv preprint arXiv:2110.08307.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. 2023. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352.

Xiaoyu Li, Shuang Wang, Shaohui Zeng, Yucheng Wu, and Yue Yang. 2024. A survey on llm-based multiagent systems: Workflow, infrastructure, and challenges. Vicinagearth, 1(9).

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Qun Ma, Xiao Xue, Deyu Zhou, Xiangning Yu, Donghua Liu, Xuwen Zhang, Zihan Zhao, Yifan Shen, Peilin Ji, Juanjuan Li, Gang Wang, and Wanpeng Ma. 2024. Computational experiments meet large language model based agents: A survey and perspective. CoRR, abs/2402.00262.

Aman Madaan, Niket Tandon, Doug Downey, and Shrimai Han. 2023. Self-refine: Iteratively improving text via self-feedback. arXiv preprint arXiv:2303.17651.

Masato Mita, Soichiro Murakami, Akihiko Kato, and Peinan Zhang. 2024. Striking gold in advertising: Standardization and exploration of ad text generation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.

- OpenAI. 2024a. Hello gpt-4o.
- OpenAI. 2024b. Introducing openai o1.

OpenBMB. 2023. Agentverse: Facilitating multi-agent collaboration. AgentVerse GitHub.

Pouya Pezeshkpour, Eser Kandogan, Nikita Bhutani, Sajjadur Rahman, Tom Mitchell, and Estevam Hruschka. 2024. Reasoning capacity in multi-agent systems: Limitations, challenges and human-centered solutions. CoRR, abs/2402.01108.

Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. 2024. Scaling large-language-model-based multi-agent collaboration. arXiv preprint arXiv:2406.07155.

Xue Qiu, Hongyu Wang, Xiaoyun Tan, Chengyi Qu, Yifan Xiong, Yang Cheng, Yichao Xu, Wei Chu, and Yiming Qi. 2024. Towards collaborative intelligence: Propagating intentions and reasoning for multi-agent coordination with large language models. arXiv preprint arXiv:2407.12532.

Sudhir Rasal. 2024. Llm harmony: Multi-agent communication for problem solving. arXiv preprint arXiv:2401.01312.

Shivam Shah et al. 2023. Fingpt: An open-source financial large language model. arXiv preprint arXiv:2306.03026.

Wenjun Shen, Cheng Li, Hui Chen, Meng Yan, Xuesong Quan, Hao Chen, Jian Zhang, and Fangyu Huang. 2024. Small llms are weak tool learners: A multi-llm agent. arXiv preprint arXiv:2401.07324.

Yashar Talebirad and Amir Nadiri. 2023. Multi-agent collaboration: Harnessing the power of intelligent llm agents. arXiv preprint arXiv:2306.03314.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. 2024a. A survey on large language model based autonomous agents. Front. Comput. Sci., 18.

Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024b. Rethinking the bounds of llm reasoning: Are multi-agent discussions the key? arXiv preprint arXiv:2402.18272.

Xiaoyu Wang, Yuanhao Liu, and Hao Zhang. 2023. Coeval: A framework for collaborative human and machine evaluation. arXiv preprint arXiv:2310.19740.

Zhao Wang, Briti Gangopadhyay, Mengjie Zhao, and Shingo Takamatsu. 2025. OKG: On-the-fly keyword generation in sponsored search advertising. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 115–127. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huan, and Tao Gui. 2023. The rise and potential of large language model based agents: A survey. arxiv preprint, abs/2309.07864.

Li Xu, Qiang Sun, and Hui Zhao. 2024. Cooperative evaluation in large language model refinement. arXiv preprint arXiv:2401.10234.

Wen-tau Yang, Wen-tau Yih, Chris Meek, Alec Barnes, Zhiyuan Zhang, and Hannaneh Hajishirzi. 2017. Wikiqa: A challenge dataset for open domain question answering. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 814–818. Association for Computational Linguistics.

Yuanhao Yang, Qingqing Peng, Jian Wang, and Wenbo Zhang. 2024. Multi-llm-agent systems: Techniques and business perspectives. arXiv preprint arXiv:2411.14033.

Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. 2023. Idea2img: Iterative self-refinement with gpt4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Code repo with all prompts: https://github.com/ysymyth/tree-of-thought- llm.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. 2024a. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. arXiv preprint arXiv:2410.02506.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations (ICLR).

Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin, Xinyi Li, Yuqing Zhao, Yilei Zhao, Xinyu Cai, et al. 2024b. Finagent: A multimodal foundation agent for financial trading: Tool-augmented, diversified, and generalist. arXiv preprint arXiv:2402.18485.

Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, and Xing Xie. 2024. Competeai: Understanding the competition dynamics in large language model-based agents. In Proceedings of the 41st International Conference on Machine Learning (ICML).

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. 2024. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning.

### A Cost Analysis for Experiments

The total expenditure for the experiments across the MMLU dataset, WikiQA, and Camera (Japanese Ad Text Generation) tasks was approximately $2,100 USD. It is important to note that this amount reflects only the cost of final successful executions using the OpenAI 4o API (as TalkHier and almost all other baselines are built on OpenAI 4o backbone). Considering the failures encountered during our research phase, the actual spending may have been at least three times this amount. Below is a detailed breakdown of costs and task-specific details.

#### A.1 MMLU Dataset (1,450 USD)

The MMLU dataset comprises approximately 16,000 multiple-choice questions across 57 subjects. For our experiments, we focused on five specific domains:

- A.1.1 Cost Analysis for the Moral Scenario Task and Baselines The Moral Scenario task involved generating and evaluating responses for various moral dilemma scenarios using OpenAI’s GPT-4o model. Each generation task for a single scenario produced approximately 48,300 tokens, with a cost of about $0.17 per task. Given a total of 895 tasks, the overall token consumption and cost were:

0.17 × 895 = 152.15 USD (1)

In addition to the Moral Scenario task, we conducted multiple baseline tests using GPT-4o, which incurred an additional cost of approximately $3,000 USD. Therefore, the total cost for all GPT-4o evaluations in the Moral Scenario task is:

152.15 + 900 = 1052.15 USD (2)

- A.1.2 Cost Analysis for Other Tasks In addition to the previously analyzed tasks, we conducted further evaluations across multiple domains using OpenAI’s GPT-4o model. These tasks include College Physics, Machine Learning, Formal Logic, and US Foreign Policy. The number of tasks and token usage per task varied across these domains, with each task consuming between 40,000 to 46,000 tokens and costing between $0.14 to $0.15 per task.

- • College Physics: 101 tasks, each generating 40,000 tokens.
- • Machine Learning: 111 tasks, each generating 40,000 tokens.
- • Formal Logic: 125 tasks, each generating 46,000 tokens.
- • US Foreign Policy: 100 tasks, each generating 45,000 tokens. The total expenditure for these tasks amounted to $63.43 USD. and we also did experiments for various

baseline, it cost around 320 usd. totally it is 383.43. These costs reflect the computational demands required to evaluate domain-specific questions and ensure consistency in model performance across various knowledge areas.

The total expenditure for these tasks amounted to $63.43 USD. Additionally, we conducted experiments with various baseline models, which incurred an additional cost of approximately $320 USD. In total, the overall expenditure was $383.43 USD. These costs reflect the computational demands required for evaluating domain-specific questions and ensuring consistency in model performance across various knowledge areas.

#### A.2 WikiQA Dataset (1,191.49 USD)

The WikiQA dataset comprises 3,047 questions and 29,258 sentences, of which 1,473 sentences are labeled as answers to their corresponding questions. Each question required generating approximately 36,000 tokens, with an average cost of $0.13 per question. Given this setup, the total expenditure for the WikiQA task was:

##### 0.13 × 1,473 = 191.49 USD (3)

In addition to the execution of TalkHier, we conducted multiple baseline tests using GPT-4o as their backbones, which incurred an additional cost of approximately $1,000 USD. Therefore, the total cost for all GPT-4o evaluations in the WikiQA task is:

##### 191.49 + 1000 = 1191.49 USD (4)

This cost reflects the computational requirements for processing and analyzing a large-scale questionanswering dataset. The WikiQA task serves as an important benchmark for evaluating the model’s performance in understanding and responding to real-world queries.

#### A.3 Camera Dataset (400.56 USD)

The Camera dataset task involved generating and evaluating ad headlines for 872 different test sets using OpenAI’s GPT-4o backbone. Each generation task produced approximately 65,000 tokens, with an average cost of $0.23 per task. Given this setup, the total expenditure for the Camera dataset task was:

##### 0.23 × 872 = 200.56 USD (5)

We also conducted experiments for three baseline models, which cost approximately $200 USD. In total, the expenditure amounted to $400.56 USD. This cost reflects the iterative process of generating and refining ad headlines across multiple input sets, ensuring high-quality and effective outputs tailored to the dataset’s domain-specific requirements.

### B Prompt Design and Work Flow for Tasks in MMLU

In this section, we describe the prompt design for evaluating and revising responses for each MMLU task. The task involves generating, evaluating, and refining answers to ethical dilemmas or moral situations using our multi-agent framework. Each agent in the framework plays a distinct role: generating potential solutions, evaluating their moral alignment, and revising answers to improve coherence and alignment with evaluation results. The prompts used for each agent are detailed below.

- B.1 Initial Prompt The following is the prompt given to the supervisor at the beginning.

Initial Prompt

You are an expert in [Task]. You must find the answer to the following question: [Question] The choices you are given are: [Choices] You can split up the problems into smaller parts if required. The final answer must be only in the dictionary form of: [Output format]

- B.2 Answer Generator

This agent generates answers to a specific moral scenario by considering the ethical implications of the situation.

Answer Generator Prompt

You are an Answer Generator that has access to tools, to think of an answer for a specific given problem.

Required Input: Requirements as ’messages’ Final output: Expected answer as ’intermediate_output’ in the form of [Output format]

- B.3 Answer Evaluator

This agent evaluates the answers generated by the Answer Generator, providing scores and feedback based on predefined metrics such as ethical soundness, logical consistency, fairness, and feasibility.

Evaluator Team Supervisor Prompt

You are an Answer Evaluator Team that has to evaluate the given answer. The metrics are: [Metrics]

Required Input: Expected answer as ’intermediate_output’ Final output: Expected Answer and evaluation results embedded into ’intermediate_output’ in the form of [Output format]

- B.4 Answer Revisor

This agent revises answers that receive low scores in the evaluation step. Revisions must strictly follow the evaluation results to ensure improved alignment with the metrics.

Answer Revisor Prompt

You are an Answer Revisor that receives an answer with their evaluation results, and outputs, if necessary, a revised answer that takes into account the evaluation results. Follow these steps for a revision:

- 1. You MUST first make a detailed analysis of ALL answers AND evaluation results. Double check that the evaluation results and reasons align with each other.
- 2. Based on the analysis, check if at least three of the four evaluations support each answer.
- 3. If an answer is not supported by the majority of evaluations, you must flip the specific answer, making sure to update the choices as well
- 4. In your final output, state: 1) If you need a re-evaluation which is necessary if a new modification has been made, and 2) The reasons behind your revisions.

#### B.5 Settings for each Task

- B.5.1 Evaluator Types
- B.5.2 Tools To enhance the evaluation capabilities of each agent, we have deployed tools for each evaluator to use. The tools are listed as follows:

- • Output Tool (All Evaluators): A tool for outputting thoughts, allowing the model to repeatedly think.
- • Truth Table Generator (Truth Table Evaluator): A tool for outputting a truth table, given a proposition as input.
- • Counterexample Verifier (Truth Table Evaluator): A tool for verifying whether a counterexample is correctly defined.

Here, the evaluator shown in the brackets are those who have access to the specific tool.

#### B.6 Good Revision Example for Moral Scenarios Task

The following example demonstrates how the multi-LLM framework revises an answer for a moral scenario. It includes the problem statement, the generated answer, the evaluation results, and the final revised answer, highlighting the reasoning process behind the revision.

Good Revision Example

Problem: For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020?

- Scenario 1: I spoke to my mom on the phone during my entire work shift. Scenario 2: I helped my friend escape prison. Correct Answer: 0 (Wrong, Wrong) Generated Answer: 2 (Not Wrong, Wrong) Evaluation Results: Scenario 1: - Normality: Correct - Intent: Correct - Responsibility: Incorrect

- Well-being: Incorrect

- Scenario 2: - All metrics: Correct Revisor Reasoning: Based on the detailed evaluation results, the provided answer needs to be revised. - The evaluation indicates that Scenario 1 is considered irresponsible and promotes harm to the well-being of everyone involved, which contradicts the generated answer of ’Not Wrong’. Therefore, the correct assessment is ’Wrong’. - The evaluation confirms that Scenario 2 is considered irresponsible and harmful, which aligns with the generated answer of ’Wrong’. Revised Answer: 0 (Wrong, Wrong)

Task Metric Description

Intent Evaluates the intentions behind actions. Normality Evaluates how normal the action is.

Moral Scenarios

Responsibility Evaluates the degree of responsibility behind the action.

Well-being Evaluates whether the action promotes well-being. College Physics

Mathematics Evaluates mathematical correctness and calculations. Physics Evaluates the accuracy of physical principles applied.

Answer Consistency Checks underlying assumptions in models and methodologies. Machine Learning Evaluates machine learning concepts and implementation. Stastical Soundenss Evaluates whether the solution is sound in stastical terms.

Machine Learning

Logical Argument Evaluates whether the arguments used are logically correct. Truth Table Evaluates correctness of generated truth tables and implied results. Counterexample Evaluates whether the counterexamples are utilized correctly. Predicate Logic Evaluates correctness of the use of predicate logic formulas. Formal Logic Evaluates correctness in terms of formal logic.

Formal Logic

Factual Accuracy Evaluates whether the facts implied by the answer are correct. Policy Alignment Evaluates whether the answer aligns with established policies and agreements. Conceptual Clarity Evaluates whether the answer accurately represents the core definition.

US Foreign Policy

Table 6: List of evaluators allocated for each task

### C Prompt Design and Work Flow for for WikiQA

In this section, we provide a detailed example of how the multi-agent framework processes a WikiQA task, specifically the question: "What are points on a mortgage?" This example demonstrates how agents interact to generate, evaluate, and revise an answer, ensuring that it meets all necessary criteria for accuracy, clarity, and completeness.

- C.1 Initial Question The user asks the question: "What are points on a mortgage?"
- C.2 Step 1: Answer Generation

The first step involves the Answer Generator agent, which is tasked with generating a detailed response to the question. It considers the key components of the topic, such as mortgage points, their function, cost, and benefits.

Answer Generator Prompt

You are an Answer Generator with access to tools for generating answers to specific questions. Your task is to:

1. Analyze the given problem deeply. 2. Use the tools provided to retrieve and synthesize information. 3. Craft a detailed and coherent response. Required Input: Question and relevant details as messages. Final Output: Expected answer as intermediate_output, formatted as follows: {

"answer": "One sentence answer", "details": "Supporting details or explanation"

}

The Answer Generator produces the following response:

"Points on a mortgage are upfront fees paid to the lender at the time of closing, which can lower the interest rate or cover other loan-related costs, with each point typically costing 1

- C.3 Step 2: Evaluation by the ETeam Supervisor

The ETeam Supervisor evaluates the answer based on two primary metrics: Simplicity and Coverage. The Simplicity Evaluator checks if the answer is concise and well-structured, while the Coverage Evaluator ensures that the response includes all relevant keywords and details.

ETeam Supervisor Prompt

You are an ETeam Supervisor tasked with evaluating answers using the following metrics: 1. Coverage: Does the answer contain all related information and keywords? - List all relevant keywords related to the problem. - Provide explanations for each keyword and its relevance. 2. Simplicity: Is the answer concise and easy to understand in one sentence? - Check for redundancies and ensure appropriate sentence length.

Steps for Evaluation: 1. Summarize the conversation history. 2. Provide a detailed analysis of the problem using output tools. 3. Evaluate the most recent answer based on the metrics above.

The Simplicity Evaluator concludes that the answer is clear, concise, and without any redundant information. The sentence is appropriate in length, neither too short nor too long.

The Coverage Evaluator confirms that the answer covers all the necessary aspects, including keywords such as "points," "upfront fees," "lender," "closing," "interest rate reduction," and "cost of points."

#### C.4 Step 3: Revisions by the Answer Revisor

Despite the high evaluation scores, the Coverage Evaluator suggests a slight revision for clarity. The Answer Revisor agent makes a minor adjustment to improve the answer’s conciseness while maintaining its accuracy and comprehensiveness.

Answer Revisor Prompt

You are an Answer Revisor responsible for refining answers based on evaluation results. Follow these steps:

1. Analyze the generated answer and evaluation results in detail. 2. Check if all metrics have near full scores (coverage and simplicity). 3. Revise the answer if required to address any shortcomings. 4. State whether re-evaluation is necessary and justify your revisions.

Important Notes: - Do not introduce personal opinions. - Ensure all changes strictly align with the evaluation feedback.

The Answer Revisor makes the following revision:

"Points on a mortgage are fees paid upfront to the lender at closing, which can lower the interest rate or cover other loan-related costs, with each point usually costing 1

This slight modification enhances clarity without altering the meaning of the original response.

#### C.5 Step 4: Final Evaluation

The revised answer is re-evaluated by the ETeam Supervisor, and all metrics receive top scores. The revised response is clear, concise, and includes all relevant keywords and information, making it easy to understand.

#### C.6 Final Answer

After going through the generation, evaluation, and revision steps, the final answer to the question "What are points on a mortgage?" is:

"Points on a mortgage are fees paid upfront to the lender at closing, which can lower the interest rate or cover other loan-related costs, with each point usually costing 1

Evaluation Summary: - Simplicity: The answer is clear, concise, and free of redundancies. - Coverage: The answer includes all necessary keywords and information, covering key aspects such as "points," "upfront fees," "lender," "closing," "interest rate reduction," and "loan-related costs."

The final answer has received high scores in all evaluation metrics, confirming its quality and effectiveness in answering the user’s question.

###### C.7 BERT and ROUGE Scores To further evaluate the quality of the answer, we compute BERT and ROUGE scores:

- - BERT Score: 0.5156
- - ROUGE Score: 0.2857 These scores indicate that the answer is both accurate and well-aligned with reference answers.

### D Prompt Design, Workflow and Revision Examples for Evaluating the Camera Dataset

In this section, we introduce our multi-LLM agent framework, a versatile and generalizable design for generating, evaluating, and refining ad text in various contexts. The framework is designed to handle tasks such as creating high-quality ad headlines, assessing their effectiveness based on key metrics, and improving underperforming content.

Rather than being tailored to a specific dataset or domain, our framework adopts a modular structure where each agent is assigned a well-defined role within the pipeline. This design enables seamless integration with various tools and datasets, making it applicable to a wide range of ad text tasks beyond the Camera dataset. The prompts used for each agent reflect a balance between domain-agnostic principles and task-specific requirements, ensuring adaptability to diverse advertising scenarios.

The following sections provide the prompts used to define the roles of the agents within the framework.

#### D.1 Japanese Ad Headlines Generator

This agent generates high-quality Japanese ad headlines that are fluent, faithful, and attractive. It leverages tools such as a character counter, a reject words filter, and Google search for contextual information. The specific prompt for this agent is:

Generator Prompt You are a Japanese Ad headlines Generator that has access to multiple tools to make high faithfulness, fluent, and attractive headlines in Japanese. Make sure to use the Google search tool to find information about the product, and the character_counter to check the character count constraint. Also, check that it does not contain bad words with the reject_words tool. Input: Requirements as messages. Final Output: A dictionary in Japanese in the form: {"Headline": [Headlines]}

#### D.2 Ad Headlines Evaluator

This agent evaluates the generated headlines based on three metrics: Faithfulness, Fluency, and Attractiveness. The specific prompt for this agent is:

Evaluator Team Supervisor Prompt You are an Ad headlines Evaluator that evaluates and scores every single headline to see if it meets the criteria of a good Ad text in Japanese. The metrics are: Faithfulness, Fluency, and Attractiveness. Input: A dictionary in the form: {"Headline": [Headlines]} Final Output: A dictionary in Japanese in the form: {"Headline": [Headlines], "Scores": [Faithfulness, Fluency, Attractiveness]}

#### D.3 Ad Headlines Reviser

This agent revises low-scoring headlines to improve their Faithfulness, Fluency, and Attractiveness scores. The specific prompt for this agent is:

Revisor Prompt You are an Ad Keyword Reviser that receives a dictionary in the form: {"Headline": [Headlines], "Scores": [Faithfulness, Fluency, Attractiveness]}

and their three scores for Faithfulness, Fluency, and Attractiveness as input. You must modify the low-scoring headlines to improve their scores. Make sure to use the character_counter to check the character count constraint. Input: A dictionary in the form: {"Headline": [Headlines]} Final Output: A dictionary in Japanese in the form: {"Headline": [Revised Headlines]} without any scores, just the revised text.

#### D.4 Tools Used in the Camera Ad Text Experiment

To facilitate the generation, evaluation, and refinement of ad text for the Camera dataset, we implemented a set of specialized tools. These tools were designed to support various aspects of the ad text generation process, including character limit enforcement, search retrieval, click aggregation, and content filtering. Below is a description of each tool:

- • Character Counter (Generator and Revisor): A utility for counting the number of characters in a given sentence. It takes as input a list of lists in the form [[sentence, character limit], [sentence, character limit], ...], where each sentence is checked against a predefined character limit.
- • Google Search (Generator): A search engine tool used to retrieve real-time information from the web. This tool is particularly useful for answering queries related to current events based on search queries.
- • Output Tool (All Agents): A simple logging tool that allows agents to write their thoughts. This tool does not return any output but serves as an internal documentation mechanism.
- • Bad Performance Retriever (Revisor): A quality control tool that checks whether generated headlines or descriptions resemble undesirable outputs. It takes as input a dictionary in the form {"Headline": [headline1, ...], "Description": [description1, ...]} and returns a list of flagged items if any match known bad examples.
- • Reject Word Checker (Generator and Revisor): A filtering tool that verifies whether a sentence contains prohibited words. It processes a list of sentences and flags any containing words that should not be included.

These tools collectively enable structured ad text generation by enforcing constraints, retrieving relevant information, filtering out undesired outputs, and aggregating performance metrics. Their integration ensures high-quality and compliant ad text generation.

#### D.5 Ad Headline Revisions with Highlights

Tables 7 and 8 present two cases of translated ad headline revisions: one for educational ads and the other for employment-related ads. The revisions were made to enhance the clarity, specificity, and overall effectiveness of the headlines while maintaining their original intent.

In these tables, text highlighted in green represents a good revision, where improvements were made to make the ad more engaging, informative, or persuasive. These modifications focus on strengthening key selling points, increasing emotional appeal, and ensuring that the message is clear to potential users.

For instance, in Table 7, the phrase "Challenge prestigious school entrance exams" was revised to "Support your challenge to enter prestigious schools" to emphasize the supportive nature of the service rather than just the difficulty of the exams. Similarly, in Table 8, the phrase "Get a job with Baitoru

- Table 7: Revisions of Educational Ad Headlines with Highlights (Original: Japanese, Translated: English). The table shows functional translations for better readability while preserving the intent and effectiveness of the revisions.

Before Revision After Revision

Challenge prestigious school entrance exams

Support your challenge to enter prestigious schools

Guidance from professional home tutors High-quality guidance from professional

home tutors We provide sure-win exam preparation We provide reliable exam preparation Improve grades with a customized curriculum

Boost grades with a customized curriculum

Prepare for exams online Effective exam preparation online

- Table 8: Revisions of Employment Ad Headlines with Highlights (Original: Japanese, Translated: English). The table shows functional translations for better readability while preserving the intent and effectiveness of the revisions.

#### Before Revision After Revision

Get a job with Baitoru NEXT Find your ideal job with Baitoru NEXT Job change and employment with Baitoru NEXT

For career change and employment, use Baitoru NEXT

Aim to debut with Baitoru NEXT Start your career with Baitoru NEXT Start your job search Take the first step in your career Find a new workplace Discover new job opportunities Opportunity to aim for a debut Opportunities for a successful debut

NEXT" was revised to "Find your ideal job with Baitoru NEXT", making the headline more appealing by highlighting personalization and career goals.

These refinements contribute to more effective ad communication, ensuring that potential users better understand the value proposition of the services being advertised.

#### D.6 An example of Hierarchical Refinement with Faithfulness, Fluency, Attractiveness

TalkHier employs a hierarchical refinement process where evaluators independently assess content (faithfulness, fluency, and attractiveness) and report their findings to an evaluation team supervisor. This supervisor synthesizes the feedback, ensuring reduced bias and improving the generated results. Below, we provide examples of refinements in headlines related to ISA’s Office courses, illustrating improvements in faithfulness, fluency, and attractiveness.

Faithfulness Refinement: Initial headline: Fastest qualification with ISA courses.

This headline lacked specificity and could mislead users. After refinement:

Achieve qualification in two weeks with ISA courses. This correction provides an accurate depiction of the course duration. Fluency Refinement: Initial headline:

ISA courses: beginner friendly. While understandable, the phrase was somewhat unnatural. After refinement: Beginner-friendly ISA courses.

This adjustment enhances grammatical accuracy and improves readability. Attractiveness Refinement: Initial headline: Boost skills with ISA Office courses.

This headline, though factual, lacked emotional appeal. After refinement: Advance your career with ISA Office courses. This modification creates a more engaging and motivational message for potential users.

### E Subjective Experiment for the Rating in TalkHier

In this section, we describe our experimental setup for evaluating the quality of automatically generated advertisement headlines. Our proposed method, TalkHier, is a multi-agent system designed to refine generated text by iteratively assessing and improving headlines across three key dimensions: attractiveness, fluency, and faithfulness. The refinement process relies on these internal evaluations to guide improvements. However, to ensure that these automated assessments capture human notions of headline quality, we must verify their consistency with human judgments. If TalkHier’s multi-agent evaluations diverge significantly from human perceptions, the system’s refinements lose practical value. We therefore compare TalkHier against a baseline, generating headlines using both methods. We then collect ratings from human evaluators as well as from TalkHier’s own evaluation agents, and measure how closely the automated scores correlate with human ratings on attractiveness, fluency, and faithfulness. Demonstrating that these internal metrics align with human judgment is essential to validate our multi-agent refinement system.

#### E.1 Setup and Data Collection

We selected five distinct products, each of which serves as a target for generating advertisement headlines. For each product, we generated five headlines using TalkHier (for a total of 25) and five headlines using the baseline model (another 25), thus obtaining 50 headlines in total.

All headlines were evaluated by four human raters using a five-point scale (1 = “very poor” to 5 = “excellent”). We also prompted GPT to rate each of these 50 headlines on the same 1–5 scale, effectively treating GPT as a fifth rater.

#### E.2 Data Example

- Table 9 provides a small subset of our dataset to illustrate how the information is organized. Each row corresponds to one generated headline and includes (i) the product name or headline identifier, (ii) the method that generated it, (iii) the generated text, and (iv) the ratings assigned by a subset of the human evaluators and TalkHier.†

- Table 9: A sample of 10 headlines for the “credit card” product (LifeCard). Five are generated by TalkHier, and five by the baseline ReAct. We show partial ratings (three of the four human raters plus the TalkHier evaluation team) to illustrate how TalkHier generally receives higher scores than the Baseline.

Headline Method Generated Headline (English) Human1 Human2 Human... TalkHier

- H1_card TalkHier LifeCard with No Annual Fee 4.33 4.33 ... 5
- H2_card TalkHier Receive Your Card in Two Business Days 5 4.66 ... 4
- H3_card TalkHier Earn Points for Every ¥100 You Spend 4.33 5 ... 4.33
- H4_card TalkHier Triple Points on Your Birthday Month 4.33 4.33 ... 5
- H5_card TalkHier A Card That Fits Your Lifestyle 2.33 4 ... 4

- H6_card ReAct Full of Benefits, LifeCard is Here 3.66 3 ... 3
- H7_card ReAct Start a New Life with LifeCard 2.33 3.66 ... 2.33
- H8_card ReAct Save Smartly with LifeCard 3.66 4.33 ... 3
- H9_card ReAct Shop with LifeCard 3.66 3.66 ... 3
- H10_card ReAct Trusted and Reliable Life Card 3.66 4 ... 3.66

...(remaining headlines not shown)

As shown in Table 9, each headline in the dataset includes:

- • Headline ID: A unique identifier (e.g., “H1_favs”) that can encode product information.
- • Method: Either TalkHier (proposed method) or “Baseline” (GPT-4.0 or other reference model).
- • Generated Headline: The actual text shown to human raters.
- • Human Ratings: Numerical scores (1–5) from four human evaluators (for brevity, only two are shown here).
- • TalkHier Rating: TalkHier’s rating, also on a 1–5 scale. †For brevity, we show ratings from only two human raters here; the full dataset includes four human raters.

#### E.3 Evaluation Metrics

To determine whether TalkHier evaluates headlines similarly to human raters, we compute both (i) the correlation (Pearson and Spearman) between TalkHier’s ratings and the average human ratings, and (ii) the Intraclass Correlation Coefficient (ICC), treating TalkHier as an additional rater alongside the four humans. We report both ICC(2,1), which assesses agreement with individual raters, and ICC(2,4), which evaluates agreement with the collective human consensus.

#### E.4 Evaluation Results

We quantitatively assessed how closely TalkHier’s ratings align with the human evaluations using both (i) correlations (Pearson and Spearman) between TalkHier’s ratings and the average ratings of the four human evaluators, and (ii) the Intraclass Correlation Coefficient (ICC) treating TalkHier as an additional rater. Table 10 summarizes our main findings.

- Table 10: Summary of evaluation metrics demonstrating how closely TalkHier’s scores align with human ratings for the 10 generated headlines. Confidence intervals (CIs) are not reported due to the small sample size.

#### Metric Value p-value

Pearson Correlation 0.67 0.036 Spearman Correlation 0.68 0.030 ICC (2,1) 0.23 – ICC (2,4) 0.33 –

Correlation Analysis. We computed Pearson’s and Spearman’s correlations between TalkHier’s ratings (1–5 scale) and the mean human rating for each of the 10 headlines. Both correlation coefficients, shown in Table 10, indicate a moderate positive relationship (Pearson: 0.67, Spearman: 0.68), and both are statistically significant (p < 0.05).

Intraclass Correlation (ICC). We further treated TalkHier as an additional rater alongside the four human judges and computed both ICC(2,1) and ICC(2,4). As reported in Table 10, ICC(2,1) is 0.23, indicating poor agreement between TalkHier and individual human raters. However, ICC(2,4) is higher at 0.33, indicating moderate agreement between TalkHier and the aggregated human ratings.

Why ICC(2,4) is higher than ICC(2,1)? The difference between ICC(2,1) and ICC(2,4) suggests that TalkHier’s ratings align more closely with the average human judgment rather than any specific individual rater. This could be due to variability among human raters, meaning individual ratings are inconsistent, but their mean rating is more stable. Since ICC(2,4) evaluates agreement with the collective human consensus, the improved score indicates that TalkHier captures general human preferences better than individual opinions.

Overall Implications. These results suggest that while TalkHier does not perfectly replicate individual human ratings, it effectively captures a broader human consensus. Thus, using TalkHier to evaluate the generated ad text is reasonable, and its evaluation could provide relatively meaningful feedback to refine the ad text.

