# arXiv:2603.03379v2[cs.IR]21Jun2026

## MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning

Jiejun Tan Zhicheng Dou∗

zstanjj@ruc.edu.cn dou@ruc.edu.cn Gaoling School of Artificial Intelligence Renmin University of China Beijing, China

Liancheng Zhang Yuyang Hu Yiruo Cheng

Gaoling School of Artificial Intelligence Renmin University of China Beijing, China

Ji-Rong Wen

jrwen@ruc.edu.cn Gaoling School of Artificial Intelligence Renmin University of China Beijing, China

#### Abstract

As Large Language Models (LLMs) are increasingly used for longduration tasks, maintaining effective long-term memory has become a critical challenge. Current methods often face a trade-off between cost and accuracy. Simple storage methods often fail to retrieve relevant information, while complex indexing methods (such as memory graphs) require heavy computation and can cause information loss. Furthermore, relying on the working LLM to process all memories is computationally expensive and slow. To address these limitations, we propose MemSifter, a novel framework that offloads the memory retrieval process to a small-scale proxy model. Instead of increasing the burden on the primary working LLM, MemSifter uses a smaller model to reason about the task before retrieving the necessary information. This approach requires no heavy computation during the indexing phase and adds minimal overhead during inference. To optimize the proxy model, we introduce a memory-specific Reinforcement Learning (RL) training paradigm. We design a task-outcome-oriented reward based on the working LLM’s actual performance in completing the task. The reward measures the actual contribution of retrieved memories by mutiple interactions with the working LLM, and discriminates retrieved rankings by stepped decreasing contributions. Additionally, we employ training techniques such as Curriculum Learning and Model Merging to improve performance. We evaluated MemSifter on eight LLM memory benchmarks, including Deep Research tasks. The results demonstrate that our method meets or exceeds the performance of existing state-of-the-art approaches in both retrieval accuracy and final task completion. MemSifter offers an efficient and scalable solution for long-term LLM memory. We have open-sourced the model weights, code, and training data to support further research. 1.

∗Corresponding author. 1Codes are available at https://github.com/plageon/MemSifter

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

#### CCS Concepts

##### • Information systems → Information retrieval.

###### ACM Reference Format:

Jiejun Tan, Zhicheng Dou, Liancheng Zhang, Yuyang Hu, Yiruo Cheng, and Ji-Rong Wen. 2018. MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 15 pages. https://doi.org/ XXXXXXX.XXXXXXX

#### 1 Introduction

With the rapid progress of Large Language Models (LLMs), they show growing potential for long-duration tasks [18, 23, 27]. However, effective memory over extended periods remains a key challenge [12]. As tasks lengthen, the accumulated interaction history soon exceeds the LLM’s context window, forcing migration of information to persistent storage [44, 57]. In this work, we focus on long-term persistent memory, rather than working memory. Working memory is transient, confined to the context window, and discarded at the end of a session [3, 24].

Following the taxonomy of Hu et al. [12], most existing work uses token-based forms to store LLM memory. The simplest, “vanilla memory,” uses a linear or flat memory bank that sequentially stores raw memory segments. At inference, the system retrieves the top-𝑘 relevant pieces to concatenate into the context window. Vanilla memory is simple but faces major drawbacks: low retrieval accuracy and poor memory utilization. To address this, research has pursued two main directions: (1) Structural Enhancement during indexing, using extra computation to build richer indexes such as graphs [29, 34, 47] or hierarchies; and (2) Contextual Expansion during inference, increasing the working LLM’s input tokens.

Structural enhancement can improve retrieval diversity and accuracy over vanilla memory, but heavy indexing (summarization, entity extraction, and graph construction) adds substantial computation, and the abstraction process discards potentially important details. Moreover, because most stored memories are never reused, the upfront indexing cost is largely wasted. Contextual expansion exploits the working LLM’s own capabilities. With precise awareness of task requirements, the LLM can read and use memories more accurately and efficiently than indexing-based methods, and its computation is demand-driven. However, since the working LLM is usually a heavy one, using it for both reading long memory contexts and task execution creates a dual burden.

To resolve this dilemma, we ask: Can we get the accuracy benefit of inference-time reasoning without adding more work to the primary LLM? We propose MemSifter, a framework that offloads memory retrieval to a specialized, lightweight proxy model. The proxy uses “reasoning-before-retrieval”: it reads the raw interaction history, analyzes the current task, and selects the critical evidence. The heavy working LLM then receives only the top-𝑘 memory segments. By avoiding complex indexes and moving retrieval reasoning away from the main generation process, MemSifter improves recall with small overhead for the working LLM.

To bridge the gap between retrieval and task outcome, we propose a new Reinforcement Learning (RL) paradigm for training MemSifter. Unlike “Think-then-Rerank” methods [21, 39, 56], which optimize isolated retrieval metrics using static relevance labels, we directly align memory proxy optimization with the working LLM’s task success. This outcome-oriented design reflects two insights: (1) Goal Alignment: As an auxiliary module, the memory proxy should be judged only by its marginal contribution to downstream performance, not by proxy metrics like Recall or Precision. (2) Label Scarcity: Obtaining fine-grained gold rankings (e.g., exact top-20 orderings) for complex reasoning tasks is impractical, making supervised learning ineffective due to sparse signals. However, training a retriever only on the working LLM’s final task score has two challenges. First, it suffers from ambiguous credit assignment: a simple success signal cannot tell whether the correct answer came from retrieved memory or the LLM’s internal knowledge. Second, a sparse scalar reward fails to capture rank sensitivity. Earlier ranks should receive more credit because they decide which evidence enters the working LLM’s context first.

To address these issues, we design a reward mechanism comprising two components. We introduce Marginal Utility Reward, which isolates the genuine utility of retrieval by measuring the performance lift relative to a “no-memory” baseline, ensuring the retriever is rewarded only for bridging actual knowledge gaps. Furthermore, to enforce ranking quality without ground-truth labels, we employ a Rank-Sensitive Reward. By evaluating the LLM on varying top-𝑘 cutoffs and applying diminishing weights (derived from DCG [14]), we incentivize the proxy model to prioritize critical evidence at the very top. Finally, to mitigate the instability inherent in RL, we employ an iterative training strategy. We incorporate warm-up supervision to address the cold-start problem, and utilize dynamic curriculum sampling alongside model merging to ensure diverse exploration and consistent performance gains.

We evaluate MemSifter on eight LLM memory datasets, spanning personal memory to long-horizon research tasks. MemSifter consistently surpasses prior state-of-the-art in both memory retrieval accuracy and end-to-end task performance. Our contributions are: (1) MemSifter, a framework that offloads memory reasoning to a lightweight proxy, enabling efficient inference-time scaling without overloading the working LLM. (2) A task-outcome-oriented RL paradigm that directly optimizes the proxy for the working LLM’s final success, tightly coupling retrieval and reasoning. (3) State-of-theart results on eight diverse benchmarks with superior efficiency, along with released code and data to support future work.

2 Related Work

- 2.1 Memory Mechanisms for LLMs

Prior LLM memory research mainly studies architectures for persistent interaction, typically grouped into token-level, parametric, and latent memory. Token-level memory stores information as external units such as text chunks or embeddings. Systems like MemGPT [26] use hierarchical paging to cope with limited context windows, while Mem0 [4] and others standardize memory CRUD operations [25, 57]. Graph-based methods like GraphRAG [8] and HippoRAG [9] build knowledge graphs for long-horizon, multihop reasoning. These approaches provide explicit, editable storage but often incur high indexing costs and adapt poorly to dynamic, task-specific retrieval [11, 15]. Recent RL-based memory agents is a special branch of token-level memory [51? ]. They train an agent or a working LLM to build and use memory across conversations. They focus more on agent self-motivated long context folding and management. In comparison, MemSifter keeps raw memory outside the working LLM and trains a separate proxy for retrieval offloading. Parametric memory encodes knowledge in model parameters. Internal methods like Character-LLM [31] fine-tune models with personas or behavioral priors, while external methods like MLP-Memory [42] inject retrieved knowledge into Transformer decoders via projection layers. They offer fast, implicit access but updating memory requires (re)training, limiting real-time use. Latent memory compresses information into intermediate representations. RazorAttention [35] preserves effective attention spans, and MemoRAG [28] creates compact global memory tokens. These are efficient but often depend on complex architectural changes or lossy compression that can harm fine-grained retrieval [41]. MemSifter uses a hybrid design that combines token-level and parametric memory. It keeps raw content in a flexible token-level store for realtime updates, while delegating memory processing and retrieval to the parametric weights of a proxy model. Unlike purely parametric memory that stores facts, this proxy stores the skill of identifying and reasoning over critical information in external storage.

- 2.2 Inference-Time Retrieval Scaling

To address the semantic limits of static embedding-based retrieval, recent work turns to adding computation at query time to improve retrieval via reasoning and generation. One direction is reasoningbefore-retrieval. Standard encoders struggle with long-range dependencies and complex implications. Methods like RGE [20] and TTE [5] use Multimodal LLMs to generate structured rationales or explanatory contexts before embedding. The O1 Embedder [49] similarly inserts a latent “thinking” phase, conditioning dense representations on explicit reasoning traces. These approaches trade extra inference time for higher semantic fidelity, treating retrieval as a reasoning-centric process. In parallel, generative reranking has become a key paradigm for final retrieval. Instead of vector similarity, these methods use LLM reasoning to directly produce relevance rankings. Techniques span pointwise scoring [19] to listwise methods that reason over candidate sets [1, 10, 36]. Some rely on frozen LLMs with advanced prompting [10, 43]), while others fine-tune models for ranking via distillation or reinforcement learning [30, 59, 60].

MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

Memory Bank: <session0> <user> Can you help me improve the wording in this personal email? </user> <assistant> Here’s a refined version of your email... </assistant> ... </session>

[Figure 1]

- <session1> <user> Thanks, ... </session> ...
- <session2> <user>... </session> ... Current Task

[Figure 2]

Memory Bank

Current Task: How much money did I raise in total through all the charity events I participated in?

- O1: <think> ...</think> <ranking> 17,16,2,33...</ranking>

- O2: <think> ...</think> <ranking> 2,17,29,8...</ranking>

...

O1

O6

Reward1 Reward2

Adv 1

Adv6

MemSifter (Actor)

...

Rank Reward

MemSifter

Working Agent

[Figure 3]

[Figure 4]

| | |
|---|---|
| | |

<ranking> 27,13,34,5, … </ranking>

Session 27

Session 13

Session 26

Token Mean Compute

Memory RL

MemSifter Inference

KL Loss

MemSifter (Ref)

O6: <think> ...

... ...

<think> To figure out the total amount of money raised through all the charity events the user participated in, the following thinking steps are required: 1. Check for revelant user memory data ... In Session 27, the user talks about working on a project at NovaTech ... In Session 28, the user asks ... </think>

Reward1

Adv6

...

Reward Loss

[Figure 5]

Answer Reward

[Figure 6]

Figure 1: Top: the RL algorithm for MemSifter. Bottom: the inference pipeline for MemSifter.

Although effective, these approaches are computationally heavy, often invoking the same large models used for the main task. MemSifter takes inspiration from both lines of work: it adopts the “reasoning” of inference-time scaling and the “listwise evaluation” of generative reranking, but distills them into a specialized, lightweight proxy. This preserves the benefits of reasoning-enhanced retrieval with minimal overhead.

- 2.3 End-to-End Retrieval Optimization

A key challenge in retrieval-augmented systems is the mismatch between pre-training objectives of retrieval (e.g., semantic similarity) and the downstream needs of generation [54]. To address this, recent work optimizes retrievers using feedback from the generator instead of static relevance labels. Most methods derive supervision from the generator’s output to fine-tune retrieval, either by maximizing the likelihood of ground-truth answers [13, 33, 48] or distilling knowledge from strong cross-encoder rerankers [40, 53]. More recent feedback-driven approaches use LLMs to annotate relevance or create “silver” data, aligning retrieval scores with generation preferences [55]. These paradigms are limited for LLM memory. Standard RAG retrieves from static corpora for factual QA, whereas autonomous LLMs must retrieve from their own episodic interaction history, which is dynamic, unstructured, and contextdependent [24, 58]. Here, relevance is defined by utility for future planning, not factual matching. LLM memory tasks involve longhorizon, multi-step reasoning with no intermediate ground truths; the only reliable signal is the sparse final outcome, making finegrained supervision infeasible.

MemSifter instead uses a task-outcome-oriented RL paradigm. Rather than relying on intermediate relevance labels or answer likelihoods, we optimize the memory proxy directly from the working LLM’s final success, capturing the true “reasoning utility” of memories in the LLM’s context without human annotation or heavy computation.

3 Methodology

- 3.1 Problem Formulation

with human users and utilizing external tools. The memory bank stores the raw interaction history 𝐻 in persistent external disk, which captures the information exchange between the LLM and the external environment—ranging from user replies in assistant scenarios to tool execution outputs in research tasks. Formally, we represent the history as a sequence of sessions 𝐻 = {𝑠1, . . .,𝑠𝑁 }. Each session 𝑠𝑖 is further composed of a sequence of interaction turns 𝑠𝑖 = {𝑡1, . . .,𝑡𝑀𝑖 }.

Let 𝑞 denote the current task, representing the latest external input. The objective of the working LLM is to generate an optimal response 𝑎. Crucially, the quality of 𝑎 hinges upon the effective utilization of the long interaction history 𝐻. However, in long-horizon scenarios, processing the full history 𝐻 is often computationally prohibitive or exceeds the LLM’s context window. To address this, we introduce a memory proxy 𝑃 to retrieve a subset of memory segments 𝑀rel relevant to 𝑞 from 𝐻. The working LLM then generates the final response 𝑎 = 𝑀(𝑞,𝑀rel) conditioned on both the current task 𝑞 and the retrieved memory segments.

#### 3.2 Memory Proxy Reasoning

MemSifter operates on the interaction history 𝐻, which is initially stored in external storage. To facilitate retrieval, we first segment 𝐻 into distinct sessions based on topic continuity, where each session represents a coherent interaction sequence regarding a specific topic or tool usage. When a benchmark provides session boundaries, we keep them; for deep research trajectories, we treat each search or reasoning step as one session. We reformat these sessions by wrapping them with unique identifiers, e.g., “<session 𝑖> ...

</session>”, enabling the proxy model to distinguish between different historical contexts.

During the loading phase, if the total token count of 𝐻 exceeds the proxy model’s context window (e.g., 128k tokens), we employ a lightweight embedding model to perform a coarse-grained prefiltering. This step ranks sessions by query-session cosine similarity and keeps the top sessions under the token budget, without a fixed similarity threshold. More filtering details are in Appendix A.

The memory proxy then performs a “Think-and-Rank” inference process on this loaded context. Given the current task 𝑞 and the formatted history, it first generates a reasoning rationale𝑡 (enclosed in “<think>” tags) to analyze dependencies. It then predicts the

To facilitate a precise formulation, we define the LLM memory task as follows. Let 𝑀 denote a working LLM, capable of interacting

Historical Interaction Information: <session0>...</session>...<session1>... Current Chat Context: I’m trying to track my online spending, can you help me set up a budgeting tool or something? You are tasked with a relevance-ranking task. Your core goal is to analyze two key inputs—the user’s historical interaction information organized by sessions and the current chat context between the user and the AI assistant—then output the top 10 most relevant sessions ranked from highest to lowest relevance. Please follow these strict guidelines to complete the task:

- 1. Input Definitions Historical Interaction Information: A collection of discrete, time-stamped sessions, each containing multiple interaction segments between the user and the AI (or other systems). Each session is identified by a session tag (e.g., <session1>, <session2>). Current Chat Context: The ongoing real-time conversation between the user and the AI assistant, including all recent messages (user’s latest questions/statements + the AI’s latest responses, if any). This context reflects the user’s current needs, concerns, or topics of interest.
- 2. Relevance Judgment Criteria To determine which sessions are most relevant, focus on the following priority factors (in descending order of importance): Topic Consistency: Whether the core topic(s) of the historical session align with the user’s current chat topic. User Need Continuity: Whether the historical session reflects recurring or unresolved user needs related to the current chat. Detail Overlap: Whether key entities, terms, or scenarios in the historical session match those in the current context. Session Completeness: How well the entire session provides context and information relevant to the current query.
- 3. Output Requirements Format: Output the session identifiers of the top 10 most relevant sessions in descending order of relevance, session ids separated by comma, and wrap them with tag <ranking></ranking>. Ensure the final output contains exactly the session identifiers from the provided “Historical Interaction Information”—do not create or modify session identifiers. Accuracy: You should think carefully about the relevance of each session based on the above criteria before ranking them. Wrap your thinking process in “<think>...</think>”. Quantity: Output exactly 10 sessions if available; if fewer than 10 sessions exist, output all available sessions in ranked order. Brief Repeat: During your reasoning process, do not repeat lengthy parts of the original message. If the original message is too long, you can omit some words in the middle by replacing them with ellipsis (...). Example Output Format: <think>Your thinking process</think><ranking>27,13,34,5,12,8,21,45,6,19</ranking> Please analyze the above two inputs and output the top 10 most relevant sessions in descending order of relevance, formatted as specified.

##### Figure 2: The prompt for the memory proxy model, describing the task in detail.

IDs of the top-𝑘 most relevant sessions, enclosed in “<ranking>” tags (see Figure 2 for the prompt). We use a fixed top-10 ranking list for fair comparison, and all methods follow the same top-𝑘 budget. Finally, the content of these top-𝑘 sessions is retrieved and concatenated with 𝑞 to form the context for the working LLM, which then generates the final response 𝑎.

#### 3.3 Task-Outcome-Oriented Reward

Retrieval-metric-based rewards work well in “Think-and-Rerank” settings but do not transfer directly to our framework. Our memory proxy’s main goal is to enhance the working LLM, so retrieval quality should be judged by its impact on final task performance, not by intrinsic retrieval metrics. Fine-grained supervision is also difficult: most tasks lack ground-truth memory annotations, and available annotations rarely provide full top-𝑘 rankings as in reranking datasets. To overcome these issues, we derive a reward signal from the working LLM’s final performance, using two designs: utility estimation via ablation and rank-sensitive rewards.

3.3.1 Marginal Utility Reward. The first design component aims to quantify the genuine utility of retrieved memory. We adopt the principle of ablation study to isolate the net contribution of memory segments. Specifically, we define 𝑠0 as the baseline performance score where the working LLM attempts the task without any retrieved memory. When the working LLM is provided with the top-𝑘 retrieved segments, the score is denoted as 𝑠𝑘. The true effectiveness of the memory retrieval is thus quantified by the performance lift 𝑠𝑘 − 𝑠0. To further capture the granularity of retrieval quality, we extend this global ablation to a progressive evaluation strategy. Instead of a single static top-𝑘 check, we evaluate the LLM’s performance across a sequence of increasing context sizes, denoted as K = {𝑘1,𝑘2, . . .,𝑘𝑁 }, as shown in Figure 3a.

Evaluating the working LLM at every possible rank position would incur prohibitive computational costs. To mitigate this, we employ two optimization strategies. First, we adopt a Fibonacci sampling sequence (e.g., {1, 2, 3, 5, . . . }) for K. This reduces inference calls from linear to logarithmic in list size while still capturing key performance inflection points at the top of the list. Second, during training, we use an efficient but capable working LLM (Qwen330B-A3B), greatly speeding up the feedback loop.

For any two consecutive evaluation tiers 𝑘𝑛−1 and 𝑘𝑛, the perfor-

mance difference Δ𝑠𝑛 = 𝑠𝑘𝑛 − 𝑠𝑘𝑛−1 explicitly represents the marginal contribution provided by the newly added memory segments

ranked between 𝑘𝑛−1 and 𝑘𝑛. A potential concern is that this incremental calculation might unfairly penalize lower-ranked memories that depend on earlier information. However, in memory-based generation, utility is inherently context-dependent: a memory’s value lies in how it complements preceding context to complete a reasoning chain. By computing the marginal lift Δ𝑠𝑛 given the top-𝑘𝑛−1 memories, our reward mechanism pushes the retriever to place foundational information first (to secure an early score 𝑠𝑘1) and complementary details next, thereby building a coherent, logically progressive context for the working LLM.

3.3.2 Rank-Sensitive Reward. The progressive ablation above measures how much information is gained. It treats all gains equally regardless of their position. For retrieval, earlier ranks should receive more credit because they define which evidence is seen first by the working LLM. This reward shapes retrieval priority. It is not a claim about the working LLM’s full attention pattern. We use a diminishing returns design, analogous to the DCG metric [14].

We posit that the reward for a marginal performance gain Δ𝑠𝑛 should be discounted based on its rank position. In other words, a performance boost achieved by the top-1 memory is rewarded

Memory RL

Reward Loss

Memory Bank: <session0> <user> Can you help me improve the wording in this personal email? </user> <assistant> Here’s a refined version of your email... </assistant> ... </session>

MemSifter (Ref)

KL Loss

Reward1 Reward2

Adv 1

- O1: <think> ...</think> <ranking> 17,16,2,33...</ranking>

- O2: <think> ...</think> <ranking> 2,17,29,8...</ranking>

O1

Token Mean Compute

Answer Reward

Adv6

...

### ... ...

...

- <session1> <user> Thanks, ... </session> ...
- <session2> <user>... </session> ... Current Task

O6

MemSifter (Actor)

Rank Reward

Adv6

O6: <think> ...

Reward6

MemSifter Inference

Session 27

<think> To figure out the total amount of money raised through all the charity events the user participated in, the following thinking steps are required: 1. Check for revelant user memory data ... In Session 27, the user talks about working on a project at NovaTech ... In Session 28, the user asks ... </think>

###### Memory Bank

<ranking> 27,13,34,5, … </ranking>

| | |
|---|---|
| | |

Session 13

Current Task: How much money did I raise in total through all the charity events I participated in?

...

Working Agent

MemSifter

Session 26

MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

Weight wn vs # Memory Sessions

0.4

0.3691

[Figure 13]

Fitted Trend Weight wi

Sys Task

Score 0

Memory

Bank

| |
|---|

0.3

wWeightn

[Figure 14]

Sys Task

S 27

- Score 1

- Score 2

0.2

0.1309

0.1131

[Figure 15]

Sys S 27 S 15

Task

[Figure 16]

0.1

0.0714

### ... ...

0.0528

MemSifter ...

0.0

[Figure 17]

Sys S 27 S 6

Task R2

Score 8

1 2 3 5 8 Number of Retrieved Memory Sessions

(a) Marginal Utility Reward

(b) Diminishing Weight 𝑤𝑛

##### Figure 3: Task-oriented reward design.

heavily, while the same boost achieved by the top-5 memory receives less credit. Mathematically, we formulate the final reward 𝑅ans as a weighted aggregation of the progressive scores:

Directly computing Equation (4) requires calculating differences step-by-step. To simplify this for efficient computation, we regroup the terms by their performance scores 𝑠𝑘𝑛. This transformation isolates the baseline term 𝑠0, which carries a coefficient of −𝐷1. Given our setting of 𝑘1 = 1, we have 𝐷1 = log1

∑︁𝑁

2 2 = 1, effectively reducing the baseline term to a simple subtraction of 𝑠0.

𝛾𝑛 · (𝑠𝑘𝑛 − 𝑠𝑘𝑛−1). (1) Following the same intuition, we use the DCG-style form:

𝑅ans =

𝑛=1

Consequently, the equation can be rewritten as a weighted sum

of the accumulated scores, 𝑅ans = −𝑠0 + 𝑛 𝑁=1 𝑤𝑛𝑠𝑘𝑛, where the weight 𝑤𝑛 represents the differential discount between consecutive tiers, as visualized in Figure 3b. Specifically, 𝑤𝑛 is defined as:

∑︁𝐾

𝑐𝑖 log2(𝑖 + 1)

, (2)

𝑅ans =

𝑖=1



where 𝛾𝑛 is a decaying weight coefficient ensuring that early ranks dominate the reward signal. By rearranging the terms and normalizing against the baseline 𝑠0, this can be reformulated into a computationally efficient form:

1 log2(𝑘𝑛+1 + 1)

1 log2(𝑘𝑛 + 1)

, if 1 ≤ 𝑛 < 𝑁; 1 log2(𝑘𝑛 + 1)

−

(5)

𝑤𝑛 =

 

, if 𝑛 = 𝑁.

∑︁𝑁

Substituting these weights back into Equation (1) gives larger weights to gains at earlier ranks (smaller 𝑘𝑛) and smaller weights to gains at later ranks. This aligns our reward mechanism with DCG’s diminishing returns property and encourages the memory proxy to place critical information early.

𝑤𝑛 · 𝑠𝑘𝑛, (3)

𝑅ans = −𝑠0 +

𝑛=1

where the coefficients 𝑤𝑛 follow the logarithmic decay of DCG (i.e., ∝ 1/log2(𝑟𝑎𝑛𝑘)). This design rewards the proxy for putting useful evidence near the top of the list.

Ideally, the standard DCG formulation scales the contribution of each individual document at rank 𝑖 by a logarithmic decay factor 1/log2(𝑖 + 1). However, as described in Sec. 3.3.1, we employ a sparse Fibonacci sampling strategy for efficiency. Consequently, each evaluation step 𝑛 introduces a batch of new memory sessions (ranging from rank 𝑘𝑛−1 + 1 to 𝑘𝑛) rather than a single entry. Since we only observe the aggregate performance lift Δ𝑠𝑛 for this entire batch, we approximate the DCG summation by applying the decay factor of the batch’s outer boundary 𝑘𝑛 to the collective gain:

#### 3.4 Optimization Protocol and Curriculum

We employ an iterative training strategy to progressively enhance the retriever’s capabilities.

Dynamic Curriculum Construction To maximize sample efficiency, we construct a dynamic training set before each iteration. We define an anchor score 𝜏 (empirically set to 0.2) to identify tasks within the model’s “zone of proximal development”—specifically, difficult tasks where the model exhibits emerging capabilities but has not yet achieved mastery. We prioritize selecting task training samples where the model’s current performance aligns most closely with 𝜏. Furthermore, during the RL rollout phase, we incorporate a dynamic sampling strategy inspired by DAPO [52] to ensure the generated trajectories maintain appropriate difficulty levels, preventing the model from overfitting to easy samples or collapsing on impossible ones.

∑︁𝑘𝑛

∑︁𝑘𝑛

1 log2(𝑘𝑛 + 1)

𝑠𝑘𝑛 − 𝑠𝑘𝑛−1 log2(𝑘𝑛 + 1)

𝑐𝑖 log2(𝑖 + 1)

𝑐𝑖 =

≈

·

.

𝑖=𝑘𝑛−1+1

𝑖=𝑘𝑛−1+1

To formalize this, we approximate the theoretical DCG reward by assigning a unified discount factor to each evaluation batch. Let 𝐷𝑛 = log 1

2(𝑘𝑛+1) represent the discount factor for the memory segments up to rank 𝑘𝑛. We approximate the total reward as the sum of discounted marginal gains:

Hybrid Reward with Cold Start Although our ultimate goal is to optimize the task-outcome-oriented score 𝑅ans, an RL based solely on outcomes can suffer instability during the early stages of training (the cold-start problem). To mitigate this and accelerate convergence, we introduce a temporary auxiliary objective. During initial warm-up steps, we utilize a small set of available memory

∑︁𝑁

𝐷𝑛 · (𝑠𝑘𝑛 − 𝑠𝑘𝑛−1), (4)

𝑅ans ≈

𝑛=1

where we define the boundary conditions 𝑘0 = 0 and 𝑠𝑘0 = 𝑠0.

ranking annotations to compute a retrieval quality metric, 𝑅𝑟𝑒𝑡 (measured by DCG). This phase is strictly designed to help the proxy model learn the correct output format and basic semantic relevance. The total reward 𝑅 is formulated as a weighted sum: 𝑅 = 𝛼·𝑅ans+𝛽·𝑅𝑟𝑒𝑡, where𝛼 and 𝛽 are hyperparameters. Tomaintain

- our label-free design, we use a reward annealing schedule: 𝛽 is high only in the first epoch and then annealed to zero, shifting the

model to purely outcome-oriented optimization on 𝑅ans.

Stabilization via Model Averaging To reduce RL training fluctuations, we apply Model Averaging at the end of each iteration. We select the top-𝑘 checkpoints by validation performance and compute the arithmetic mean of their trainable parameters. The merged model initializes the next iteration, consolidating learned capabilities and smoothing optimization.

4 Experiments

- 4.1 Datasets

We select five representative personal LLM datasets (1~5) and three deep research (6~8) to evaluate our method and baselines:

(1) LoCoMo [22]: A benchmark for evaluating very long-term conversational memory, featuring 10 multimodal dialogues averaging 300 turns to assess understanding of facts and temporal relationships; (2) LongMemEval [44]: A comprehensive benchmark designed to evaluate five core long-term memory capabilities in continuous chatbot interactions; (3) PersonaMem [16]: A dataset with over 180 curated personas that assesses an LLM’s ability to internalize user profiles from long interaction histories; (4) PerMV2 [17]: A scaled-up version simulating 1,000 user scenarios, focusing on enabling models to implicitly infer preferences from long conversations; (5) ZH4O [7]: A QA dataset integrating semantic and episodic memories, covering diverse long-term memory types in mixed-context scenarios; (6) HotpotQA [50]: A dataset requiring multi-hop reasoning across multiple supporting documents to answer complex natural language questions; (7) WebWalker [46]: A benchmark assessing the capacity of agents to systematically traverse website subpages and extract multi-layered, high-quality information; (8) WebDancer [45]: A dataset focused on autonomous multi-step research, providing rich browsing trajectories for training agents in in-depth information seeking tasks.

In our experiments, we randomly sample 400 questions from the test set of the LoCoMo, PersonaMem, PersonaMem-v2, and PerLTQA datasets for evaluation. For LongMemEval, we randomly sample 150 questions for the test set. Detailed train/test split statistics are provided in Appendix A. We report end-to-end generation F1 as the unified metric across all benchmarks because every method ultimately produces a free-form answer from the retrieved memory context. For PersonaMem, we additionally use option-normalized multiple-choice accuracy; detailed accuracy results are reported in Appendix A. Retrieval quality is evaluated with NDCG@1 and NDCG@5, and efficiency is measured by input/output token counts and latency.

- 4.2 Baselines

We compare MemSifter against state-of-the-art baselines across five distinct categories. Embedding-based retrieval: (1) BGE-M3 [2]: A unified retrieval model supporting dense, sparse, and multi-vector

representations, capable of handling long inputs up to 8,192 tokens. (2) EmbeddingGemma [38]: A lightweight embedding model distilled from the Gemma family, optimizing semantic representation through geometric alignment for low-latency retrieval. LLM Memory Frameworks:(1) Mem0 [4]: A long-term memory management layer that proactively extracts, stores, and updates salient information as discrete memory units. (2) Nemori [25]: A cognitively inspired architecture that segments interaction history into coherent units and dynamically updates them via a self-organizing mechanism. (3) MemAgent [51]: An RL-based memory agent that trains the working LLM to maintain and exploit memory across multiple conversations. (4) Mem-𝛼 [? ]: An RL-based memory-construction agent that learns when and how to write structured memories from long interaction histories. Graph Retrieval: (1) HippoRAG [9]: A neuro-symbolic retrieval framework that constructs knowledge graphs from documents and utilizes Personalized PageRank to enable multi-hop associative recall. (2) A-MEM [47]: An evolving memory system that structures summarized experiences into a dynamic graph to support reasoning over complex LLM histories. Generative Rerankers: (1) Rearank [56]: An RL-based listwise reranking LLM that explicitly reasons about candidate relevance to optimize the final ranking. (2) ReasonRank [21]: A reasoningenhanced reranker trained via supervised and reinforcement learning to generate explanation-aware relevance scores. For fairness, Rearank and ReasonRank are retrained on the same training pool as MemSifter using their original single-round RL reward designs. MemAgent and Mem-𝛼 are adapted as RL-memory baselines under the same persistent-memory benchmark interface: each method constructs or maintains memory from the historical sessions, and the working LLM answers with the resulting memory context under the same top-𝑘 budget. Long-context LLMs (Retrieval-Free): (1) Qwen3-30B [37]: A large-scale Mixture-of-Experts model leveraging a massive context window to directly process full interaction histories without external retrieval. (2) DeepSeek-V3.2 [6]: A highcapacity model optimized for long-context efficiency, serving as a strong baseline for native in-context learning capabilities.

#### 4.3 Main Results

To verify the effectiveness of MemSifter, we evaluate our method against the baseline methods on the eight aforementioned datasets. Table 1 presents the main F1 results of the end-to-end experiments, with adapted RL-memory baselines included as explicit comparison rows. The proxy is trained with feedback from Qwen3-30B-A3BInstruct and evaluated with both Qwen3-30B-A3B-Instruct and DeepSeek V3.2, which tests transfer across working LLMs. Based on the experimental results, we draw the following analysis. (1) Superiority over Embedding: Compared to standard embedding retrieval methods, MemSifter achieves significantly higher retrieval precision. By filtering out irrelevant noise that naive embeddings often capture, our method provides the working LLM with a cleaner context, directly translating to substantial gains in final task success rates. (2) Task Utility vs. Generative Rerank: MemSifter outperforms “Think-then-Rerank” baselines. This confirms a critical hypothesis: semantic relevance does not equal task utility. While rerankers optimize for query-document similarity, our outcomeoriented reward ensures that the retrieved memory specifically aids

##### Table 1: Main Results of MemSifter evaluated by F1 score. The best and second best of each model are in bold and underlined.

LoCoMo LongMemEval PersonaMem PerM-V2 ZH4O HotpotQA WebWalker WebDancer

Method

32K 128K 1M 32K 128K 128K 128K 128K 128K 128K

BGE-M3 29.49 31.62 29.91 20.76 18.83 21.45 45.38 20.79 22.61 33.89 GemmaEmb 29.90 32.62 30.83 20.96 18.22 22.95 41.84 21.01 21.69 37.82 Nemori 27.43 34.39 29.24 21.25 19.89 23.72 45.07 22.76 24.39 35.28 Mem0 31.62 33.93 28.20 22.33 20.63 21.63 47.19 20.35 23.46 32.58 MemAgent 37.36 35.16 31.60 20.62 19.84 21.74 41.15 24.09 24.05 34.10 Mem-𝛼 38.70 34.71 33.07 23.37 20.27 21.01 47.27 23.02 24.49 34.58 HippoRAG 28.11 33.64 30.83 22.65 19.24 22.94 46.56 23.20 24.19 33.26 A-MEM 31.82 31.96 32.49 21.44 18.14 23.04 45.69 23.12 21.88 36.72 Rearank 33.04 33.53 32.47 22.23 18.41 19.76 47.14 19.76 24.19 37.23 ReasonRank 31.94 34.31 28.93 21.60 20.74 22.27 45.55 22.25 22.79 32.95 DS-V3.2 30.97 35.15 30.89 21.69 20.15 21.55 46.73 19.88 22.49 34.31 MemSifter 41.79 35.38 33.32 23.70 21.14 23.57 48.13 24.95 26.11 38.21

DeepSeekV3.2

BGE-M3 38.73 41.75 49.07 24.98 22.73 20.21 48.31 20.49 25.01 31.85 GemmaEmb 36.25 43.04 48.16 23.30 22.36 20.12 50.25 19.84 24.51 32.45 Nemori 40.96 46.09 47.31 24.35 22.69 18.62 45.53 20.91 25.80 32.77 Mem0 36.20 38.56 48.53 24.87 22.51 21.11 48.40 20.57 24.24 32.88 MemAgent 42.90 45.95 48.25 26.08 22.35 20.47 52.45 19.80 27.04 32.25 Mem-𝛼 40.63 45.86 44.45 24.04 22.66 22.72 48.11 22.08 27.06 30.74 HippoRAG 41.94 40.13 47.62 23.35 21.31 20.62 48.36 21.41 25.16 31.79 A-MEM 35.87 40.46 46.77 24.57 20.78 21.07 44.52 19.85 26.13 33.67 Rearank 39.19 39.55 45.60 22.37 22.95 22.17 49.02 19.95 24.14 31.40 ReasonRank 41.04 44.26 48.20 23.78 20.18 19.05 47.16 22.37 25.72 30.00 Qwen3-30B 39.81 42.20 46.42 24.25 20.82 24.49 51.68 21.59 25.25 33.63 MemSifter 46.39 47.26 49.58 26.45 23.75 22.81 50.91 22.71 27.44 35.10

Qwen3-30B-A3B-Ins

in solving the reasoning problem at hand. (3) Efficiency against Complex Architectures: Unlike complex memory pipelines (e.g., graph-based or generative memory) that incur high latency and indexing overheads, MemSifter achieves state-of-the-art performance with a lightweight architecture. This demonstrates that a well-optimized, small-scale proxy can surpass heavy, complex systems in practical scenarios. (4) Effective Alternative to LongContext Models: Compared to directly feeding full history into long-context LLMs, MemSifter reduces computational costs while maintaining or even exceeding performance. Long-context models often struggle with the “lost-in-the-middle” phenomenon; our method mitigates this by proactively selecting and reasoning about critical information before the working LLM processes it.

#### 4.4 Ablation Study

To assess the effectiveness of our reward mechanism’s core components, we run ablations by removing individual design elements.

- (1) w/o Task-Outcome Reward: We remove RL-based optimization and train the proxy only on retrieval-metric supervision (as in warm-up). The large performance drop shows that optimizing static relevance (e.g., semantic similarity) is insufficient for downstream utility: without outcome-based signals, the proxy misses memories that are semantically distant but logically crucial for LLM reasoning.
- (2) w/o Marginal Utility: We replace the incremental gain metric with the absolute task score from using top-𝑘 memories, omitting the “no-memory” baseline. This hurts performance due to ambigu-

- ous credit assignment: the proxy gets high rewards even on easy tasks solved by parametric knowledge alone, obscuring the true

contribution of retrieved information and destabilizing training. (3) w/o Rank-Sensitive Weights: We replace the DCG-weighted sum with a simple mean over top-𝑘 evaluations. The resulting degradation shows the value of rank sensitivity: uniform averaging ignores position and does not reward the proxy for placing key evidence early. (4) w/o Thinking / Direct Rank: We add a controlled direct-rank setting to isolate the role of explicit reasoning before retrieval. The result row is included in Table 3.

#### 4.5 Further Analysis

- 4.5.1 Retrieval Quality Analysis. To validate that our end-toend gains stem from superior context quality, we conduct a focused evaluation on intrinsic retrieval metrics. Figure 2 details the comparative performance. Note that due to the lack of ground-truth ranking annotations in the Deep Research dataset, this analysis is restricted to benchmarks where gold labels are available.

We find MemSifter consistently achieves higher recall and ranking precision compared to embedding-based, hierarchical, and even reasoning-heavy retrieval baselines. By accurately filtering noise and prioritizing key evidence, our proxy ensures the working LLM receives the most relevant context. This high-quality retrieval acts as the foundational guarantee for the superior downstream performance reported in the previous section, confirming that precise memory recall is a prerequisite for effective reasoning.

- 4.5.2 Training Dynamics and Curriculum Analysis. To assess our reward mechanism and iterative strategy, we plot the reinforcement learning trajectories in Figure 4. The grey curve is the baseline

##### Table 2: Main Memory Retrieval Results of MemSifter. The best and second best of each model are in bold and underlined.

LoCoMo LongMemEval PersonaMem PerM-V2 ZH4O

Method

32K 128K 1M 128K 128K 128K Metric ndcg@1 ndcg@5 ndcg@1 ndcg@5 ndcg@1 ndcg@5 ndcg@1 ndcg@5 ndcg@1 ndcg@5 ndcg@1 ndcg@5

BGE-M3 36.25 58.13 56.00 87.90 38.67 67.07 28.25 52.34 12.00 18.95 25.75 39.06 GemmaEmb 39.64 70.76 60.51 89.64 50.83 70.78 37.07 54.72 12.18 19.42 28.75 41.14 ReaRank 43.18 59.13 62.70 88.50 39.33 68.19 33.86 55.74 13.68 20.76 25.43 40.37 ReasonRank 47.64 64.94 60.81 88.18 45.14 69.32 30.59 55.18 11.26 19.94 24.49 41.34 MemSifter 70.00 78.11 67.33 89.67 60.00 75.63 43.50 61.70 15.00 20.09 40.50 50.97

##### Table 3: Ablation Study of MemSifter with DeepSeek V3.2. The best and second best of each dataset are in bold and underlined.

LoCoMo LongMemEval PersonaMem PerM-V2 ZH4O HotpotQA WebDancer WebWalker

Method

32K 1M 128K 128K 128K 128K 128K 128K

MemSifter 41.79 33.32 21.14 23.57 48.13 24.95 26.11 38.21 w/o Outcome Reward 30.59 (26.80%↓) 28.67 (13.96%↓) 15.27 (27.77%↓) 18.26 (22.53%↓) 40.05 (16.79%↓) 21.33 (14.51%↓) 20.03 (23.29%↓) 30.16 (21.07%↓) w/o Marginal Utility 38.62 (7.59%↓) 32.67 (1.95%↓) 18.66 (11.73%↓) 21.03 (10.78%↓) 46.74 (2.89%↓) 23.60 (5.41%↓) 24.70 (5.40%↓) 35.96 (5.89%↓) w/o Reward Shaping 40.68 (2.66%↓) 33.26 (0.18%↓) 19.89 (5.91%↓) 22.42 (4.88%↓) 47.30 (1.72%↓) 24.87 (0.32%↓) 25.12 (3.79%↓) 37.53 (1.78%↓) w/o Thinking 35.34 (15.43%↓) 29.81 (10.53%↓) 16.14 (23.65%↓) 19.07 (19.09%↓) 43.36 (9.91%↓) 20.69 (17.07%↓) 21.31 (18.38%↓) 32.28 (15.52%↓)

40

LoCoMoF1Score

36

32

- Epoch 1

- Epoch 2

Epoch 3 Rearank

28

24

LongMemEvalF1Score

30

25

20

- Epoch 1

- Epoch 2

Epoch 3 Rearank

15

10

0 25 50 75 100 125 150 175 200 225 Cumulative Steps (Stitched)

##### Figure 4: RL training curves of MemSifter compared with Rearank on two datasets (Datasets Aligned).

(Rearank), and the colored segments show the three stages of MemSifter’s iterative training. We conduct this analysis on LoCoMo and LongMemEval, which offer ground-truth rankings for baseline comparison. Two key observations validate our design:

- (1) Superior Alignment via Task-Outcome Reward: MemSifter converges faster and achieves higher final performance than the baseline. While Rearank relies on static relevance labels, our Task-Outcome-Oriented Reward directly aligns the proxy with the working LLM’s end-to-end success. The steeper learning curve shows that rewarding the proxy for selecting memories that improve reasoning (via rank-sensitive gradients) is more effective than optimizing semantic similarity alone.
- (2) Breaking Plateaus with Curriculum Learning: The baseline plateaus or degrades after 60–70 epochs (grey trajectory), indicating it has exhausted the signal in static data. In contrast, MemSifter continues to improve across three phases. By continually

##### Table 4: Analysis of inference cost on WebDancer(128K).

Method # Params # In-Tok # Out-Tok Latency BGE 0.2B 128K 61.08 1015.05ms ReaRank 7B 128K 1531.41 7657.05ms ReasonRank 7B 128K 1664.48 8322.40ms MemSifter 4B 128K 1557.01 3982.53ms DeepSeek-V3.2 632B 128K 712.48 49873.60ms DeepSeek-V3.2 632B 2K 705.61 759.89ms

refreshing training data to match the model’s evolving ability (emphasizing samples near the difficulty anchor 𝜏) and using model merging, we mitigate overfitting and achieves a higher upper bound.

- 4.5.3 Efficiency Analysis. We assess efficiency by normalizing input/output costs into 7B-equivalent token counts based on model size and by measuring single-H20 GPU latency (Table 4). Training cost is reported with the same accounting fields in Appendix A. The results show MemSifter’s key trade-offs: (1) Relative to lightweight embedding baselines, it adds moderate latency from generation, offset by large gains in retrieval precision. (2) Compared to “reason-then-rerank” methods, it achieves similar computational cost with better task performance. (3) Most importantly, versus Long-Context LLMs over full histories, MemSifter dramatically cuts overhead by delegating memory sifting to a smaller proxy, avoiding the heavy cost of large models on long sequences (128K→2K).
- 5 Conclusion and Future Work

This paper tackles the trade-off between retrieval precision and computational efficiency in LLM memory systems. We introduce MemSifter, a framework that separates memory management from core reasoning by offloading historical interaction “sifting” to a lightweight proxy. Using a task-outcome-based reward—combining

marginal utility and rank sensitivity, the proxy learns retrieval strategies from final task success alone, avoiding costly annotations. Experiments show MemSifter surpasses state-of-the-art baselines in task accuracy and inference efficiency. Our results present a scalable paradigm for long-horizon LLMs, where small specialized models curate context for large generalist reasoners. Future work will extend outcome-driven optimization to LLM memory consolidation and multi-modal histories.

#### References

- [1] Francesco Busolin, Claudio Lucchese, Franco Maria Nardini, Salvatore Orlando, Raffaele Perego, Salvatore Trani, and Alberto Veneri. 2025. Efficient Re-ranking with Cross-encoders via Early Exit. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2025, Padua, Italy, July 13-18, 2025, Nicola Ferro, Maria Maistro, Gabriella Pasi, Omar Alonso, Andrew Trotman, and Suzan Verberne (Eds.). ACM, 2534–2544. doi:10.1145/3726302.3729962
- [2] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.

2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 2318–2335. doi:10.18653/V1/ 2024.FINDINGS-ACL.137

- [3] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting Language Models to Compress Contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 3829–3846. doi:10.18653/V1/2023.EMNLPMAIN.232
- [4] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav.

2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. CoRR abs/2504.19413 (2025). arXiv:2504.19413 doi:10.48550/ARXIV. 2504.19413

- [5] Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Kumar Mishra, Yonghuan Yang, Jun Xiao, Qi Guo, Ser-Nam Lim, Aashu Singh, and Xiangjun Fan. 2025. Think Then Embed: Generative Context Improves Multimodal Embedding. CoRR abs/2510.05014 (2025). arXiv:2510.05014 doi:10.48550/ARXIV.2510.05014
- [6] DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bing-Li Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Hao sheng Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huan Cao, Huazuo Gao, Huixian Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, JingChang Chen, Jingting Xiang, Jingyang Yuan, Jing Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Jun-Mei Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Ling-Li Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y. Xu, Mingchuan Zhang, Minghua Zhang, Min Tang, Mingxu Zhou, P. Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiaoling Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yingpu Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yi Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Zehui Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhen guo Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang

- Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zi-Rui Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu,

Meng Li, Ning Tian, R. J. Chen, Ruiqi Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyu Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yao Li, Yi Zheng, Yuchen Zhu, Yunxiang Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dong-Li Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shao-Kang Wu, Shengfeng Ye, T.Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Ze-Na Zhang, Zhenghua Ju, Zhen Zhang, and Zihua Qu. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. https://api.semanticscholar.org/CorpusID:283448719

- [7] Yiming Du, Hongru Wang, Zhengyi Zhao, Bin Liang, Baojun Wang, Wanjun Zhong, Zezhong Wang, and Kam-Fai Wong. 2024. PerLTQA: A Personal LongTerm Memory Dataset for Memory Classification, Retrieval, and Synthesis in Question Answering. CoRR abs/2402.16288 (2024). arXiv:2402.16288 doi:10.48550/ ARXIV.2402.16288
- [8] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. CoRR abs/2404.16130 (2024). arXiv:2404.16130 doi:10.48550/ARXIV.2404.16130
- [9] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.

2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (Eds.). http://papers.nips.cc/paper_files/paper/2024/hash/ 6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html

- [10] Chi Hu, Yuan Ge, Xiangnan Ma, Hang Cao, Qiang Li, Yonghua Yang, Tong Xiao, and Jingbo Zhu. 2024. RankPrompt: Step-by-Step Comparisons Make Language Models Better Reasoners. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy, Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (Eds.). ELRA and ICCL, 13524–13536. https://aclanthology.org/2024.lrec-main.1183
- [11] Yuyang Hu, Jiongnan Liu, Jiejun Tan, Yutao Zhu, and Zhicheng Dou. 2026. Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning. (2026). arXiv:2601.04726 [cs.AI] https://arxiv.org/abs/2601.04726
- [12] Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo, Xinlei Yu, Zhenhong Zhou, Zewen Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei Yin, Xiaobin Hu, Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei Cheng, Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong Wen, Xuanjing Huang, Yu-Gang Jiang, and Shuicheng Yan. 2025. Memory in the Age of AI Agents. arXiv:2512.13564 [cs.CL] https://arxiv.org/abs/2512.13564
- [13] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.

2023. Atlas: Few-shot Learning with Retrieval Augmented Language Models. J. Mach. Learn. Res. 24 (2023), 251:1–251:43. https://jmlr.org/papers/v24/230037.html

- [14] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of IR techniques. ACM Trans. Inf. Syst. 20, 4 (2002), 422–446. doi:10.1145/582415. 582418
- [15] Shuo Ji, Mingzhe Liu, Leilei Sun, Chuanren Liu, and Tongyu Zhu. 2024. MemMap: An Adaptive and Latent Memory Structure for Dynamic Graph Learning. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2024, Barcelona, Spain, August 25-29, 2024, Ricardo Baeza-Yates and Francesco Bonchi (Eds.). ACM, 1257–1268. doi:10.1145/3637528.3672060
- [16] Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle H. Ungar, Camillo J. Taylor, and Dan Roth. 2025. Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. CoRR abs/2504.14225 (2025). arXiv:2504.14225 doi:10.48550/ARXIV.2504. 14225
- [17] Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Greg Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and Camillo Jose Taylor.

2025. PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory. https://api.semanticscholar.org/CorpusID: 283693901

- [18] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-world Github Issues?. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=VTF8yNQM66
- [19] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson,

Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu

- Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert Yüksekgönül, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda.

2022. Holistic Evaluation of Language Models. CoRR abs/2211.09110 (2022). arXiv:2211.09110 doi:10.48550/ARXIV.2211.09110

- [20] Chunxu Liu, Jiyuan Yang, Ruopeng Gao, Yuhan Zhu, Feng Zhu, Rui Zhao, and Limin Wang. 2025. Reasoning Guided Embeddings: Leveraging MLLM Reasoning for Improved Multimodal Retrieval. https://api.semanticscholar.org/CorpusID: 283109952
- [21] Wenhan Liu, Xinyu Ma, Weiwei Sun, Yutao Zhu, Yuchen Li, Dawei Yin, and Zhicheng Dou. 2025. ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability. CoRR abs/2508.07050 (2025). arXiv:2508.07050 doi:10.48550/ ARXIV.2508.07050
- [22] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 13851–13870. doi:10.18653/V1/2024.ACLLONG.747
- [23] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: a benchmark for General AI Assistants. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=fibxvahvs3
- [24] Ali Modarressi, Abdullatif Köksal, Ayyoob Imani, Mohsen Fayyaz, and Hinrich Schütze. 2025. MemLLM: Finetuning LLMs to Use Explicit Read-Write Memory. Trans. Mach. Learn. Res. 2025 (2025). https://openreview.net/forum?id= dghM7sOudh
- [25] Jiayan Nan, Wenquan Ma, Wenlong Wu, and Yize Chen. 2025. Nemori: SelfOrganizing Agent Memory Inspired by Cognitive Science. CoRR abs/2508.03341

(2025). arXiv:2508.03341 doi:10.48550/ARXIV.2508.03341

- [26] Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as Operating Systems. CoRR abs/2310.08560 (2023). arXiv:2310.08560 doi:10.48550/ARXIV.2310.08560
- [27] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Daron Anderson, Tung Nguyen, Mobeen Mahmood, Fiona Feng, Steven Y. Feng, Haoran Zhao, Michael Yu, Varun Gangal, Chelsea Zou, Zihan Wang, Jessica P. Wang, Pawan Kumar, Oleksandr Pokutnyi, Robert Gerbicz, Serguei Popov, John-Clark Levin, Mstyslav Kazakov, Johannes Schmitt, Geoff Galgon, Alvaro Sanchez, Yongki Lee, Will Yeadon, Scott Sauers, Marc Roth, Chidozie Agu, Søren Riis, Fabian Giska, Saiteja Utpala, Zachary Giboney, Gashaw M. Goshu, Joan of Arc Xavier, Sarah-Jane Crowson, Mohinder Maheshbhai Naiya, Noah Burns, Lennart Finke, Zerui Cheng, Hyunwoo Park, Francesco Fournier-Facio, John Wydallis, Mark Nandor, Ankit Singh, Tim Gehrunger, Jiaqi Cai, Ben McCarty, Darling Duclosel, Jungbae Nam, Jennifer Zampese, Ryan G. Hoerr, Aras Bacho, Gautier Abou Loume, Abdallah Galal, Hangrui Cao, Alexis C. Garretson, Damien Sileo, Qiuyu Ren, Doru Cojoc, Pavel Arkhipov, Usman Qazi, Lianghui Li, Sumeet Motwani, Christian Schröder de Witt, Edwin Taylor, Johannes Veith, Eric Singer, Taylor D. Hartman, Paolo Rissone, Jaehyeok Jin, Jack Wei Lun Shi, Chris G. Willcocks, Joshua Robinson, Aleksandar Mikov, Ameya Prabhu, Longke Tang, Xavier Alapont, Justine Leon Uro, Kevin Zhou, Emily de Oliveira Santos, Andrey Pupasov Maksimov, Edward Vendrow, Kengo Zenitani, Julien Guillod, Yuqi Li, Joshua Vendrow, Vladyslav Kuchkin, and Ng Ze-An. 2025. Humanity’s Last Exam. CoRR abs/2501.14249

(2025). arXiv:2501.14249 doi:10.48550/ARXIV.2501.14249

- [28] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025. MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation. In Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025, Guodong Long, Michale Blumestein, Yi Chang, Liane Lewin-Eytan, Zi Helen Huang, and Elad Yom-Tov (Eds.). ACM, 2366–2377. doi:10.1145/3696410.3714805
- [29] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A Temporal Knowledge Graph Architecture for Agent Memory. CoRR abs/2501.13956 (2025). arXiv:2501.13956 doi:10.48550/ARXIV.2501.13956
- [30] Ferdinand Schlatt, Maik Fröbe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Benno Stein, Martin Potthast, and Matthias Hagen. 2025. Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-ranking. In Advances in Information Retrieval - 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6-10, 2025, Proceedings, Part III (Lecture Notes in Computer Science, Vol. 15574), Claudia Hauff, Craig Macdonald, Dietmar Jannach, Gabriella Kazai, Franco Maria Nardini, Fabio Pinelli, Fabrizio Silvestri, and Nicola Tonellotto (Eds.). Springer, 323–334. doi:10.1007/978-3-031-88714-7_31

- [31] Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu. 2023. Character-LLM: A Trainable Agent for Role-Playing. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 13153–13187. doi:10.18653/V1/2023.EMNLP-MAIN.814
- [32] Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. 2025. DeepSeekMath-V2: Towards SelfVerifiable Mathematical Reasoning. CoRR abs/2511.22570 (2025). arXiv:2511.22570 doi:10.48550/ARXIV.2511.22570
- [33] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented Black-Box Language Models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (Eds.). Association for Computational Linguistics, 8371–8384. doi:10.18653/V1/2024.NAACLLONG.463
- [34] Haoran Sun and Shaoning Zeng. 2025. Hierarchical Memory for HighEfficiency Long-Term Reasoning in LLM Agents. CoRR abs/2507.22925 (2025). arXiv:2507.22925 doi:10.48550/ARXIV.2507.22925
- [35] Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning Ke, Shikuan Hong, Yiwu Yao, and Gongyi Wang. 2025. RazorAttention: Efficient KV Cache Compression Through Retrieval Heads. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https: //openreview.net/forum?id=tkiZQlL04w
- [36] Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. 2024. Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, Kevin Duh, Helena Gómez-Adorno, and Steven Bethard (Eds.). Association for Computational Linguistics, 2327–2340. doi:10.18653/V1/2024.NAACLLONG.129
- [37] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https: //arxiv.org/abs/2505.09388
- [38] Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Ju-yeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divyashree Sreepathihalli, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Qin Yin, Yun-Hsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, and Mojtaba Seyedhosseini. 2025. EmbeddingGemma: Powerful and Lightweight Text Representations. CoRR abs/2509.20354 (2025). arXiv:2509.20354 doi:10.48550/ARXIV.2509.20354
- [39] Supriti Vijay, Aman Priyanshu, Anu Vellore, Baturay Saglam, and Amin Karbasi.

2025. Think Before You Retrieve: Learning Test-Time Adaptive Search with Small Language Models. arXiv:2511.07581 [cs.AI] https://arxiv.org/abs/2511.07581

- [40] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2023. SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, 2244–2258. doi:10.18653/V1/2023.ACL-LONG.125
- [41] Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, and Julian J. McAuley.

2024. MEMORYLLM: Towards Self-Updatable Large Language Models. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https://openreview.net/forum?id=p0lKWzdikQ

- [42] Rubin Wei, Jiaqi Cao, Jiarui Wang, Jushi Kai, Qipeng Guo, Bowen Zhou, and Zhouhan Lin. 2025. MLP Memory: A Retriever-Pretrained Memory for Large Language Models. https://api.semanticscholar.org/CorpusID:281658735
- [43] Juan Wisznia, Cecilia Bolaños, Juan Tollo, Giovanni Marraffini, Agustín Gianolini, Noe Hsueh, and Luciano Del Corro. 2025. Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching. In Proceedings of the 63rd Annual Meeting of the Association for Computational

- Linguistics (Volume 2: Short Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 1064–1072. doi:10. 18653/V1/2025.ACL-SHORT.83
- [44] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu.

2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https://openreview.net/ forum?id=pZiyCaVuti

- [45] Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. WebDancer: Towards Autonomous Information Seeking Agency. CoRR abs/2505.22648 (2025). arXiv:2505.22648 doi:10.48550/ARXIV.2505.22648
- [46] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. WebWalker: Benchmarking LLMs in Web Traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 10290–10305. https://aclanthology.org/2025.acl-long.508/
- [47] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang.

2025. A-MEM: Agentic Memory for LLM Agents. CoRR abs/2502.12110 (2025). arXiv:2502.12110 doi:10.48550/ARXIV.2502.12110

- [48] Yilong Xu, Jinhua Gao, Xiaoming Yu, Yuanhai Xue, Baolong Bi, Huawei Shen, and Xueqi Cheng. 2025. Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, 629–648. doi:10.18653/V1/2025.EMNLP-MAIN.33
- [49] Ruiran Yan, Zheng Liu, and Defu Lian. 2025. O1 Embedder: Let Retrievers Think Before Action. CoRR abs/2502.07555 (2025). arXiv:2502.07555 doi:10.48550/ARXIV. 2502.07555
- [50] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, 2369–2380. doi:10.18653/V1/D18-1259
- [51] Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. 2025. MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. CoRR abs/2507.02259 (2025). arXiv:2507.02259 doi:10.48550/ARXIV.2507. 02259
- [52] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. 2025. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. CoRR abs/2503.14476 (2025). arXiv:2503.14476 doi:10.48550/ARXIV.2503.14476
- [53] Hamed Zamani and Michael Bendersky. 2024. Stochastic RAG: End-to-End Retrieval-Augmented Generation through Expected Utility Maximization. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, Grace Hui Yang, Hongning Wang, Sam Han, Claudia Hauff, Guido Zuccon, and Yi Zhang (Eds.). ACM, 2641–2646. doi:10.1145/3626772.3657923
- [54] Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, andXueqi Cheng. 2025. LLM-Specific Utility: ANew Perspective forRetrievalAugmented Generation. CoRR abs/2510.11358 (2025). arXiv:2510.11358 doi:10. 48550/ARXIV.2510.11358
- [55] Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, andXueqi Cheng. 2025. LLM-Specific Utility: ANew Perspective forRetrievalAugmented Generation. CoRR abs/2510.11358 (2025). arXiv:2510.11358 doi:10. 48550/ARXIV.2510.11358
- [56] Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and Aishwarya Agrawal. 2025. REARANK: Reasoning Re-ranking Agent via Reinforcement Learning. CoRR abs/2505.20046 (2025). arXiv:2505.20046 doi:10.48550/ARXIV.2505.20046
- [57] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing Large Language Models with Long-Term Memory. In ThirtyEighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 19724–19731. doi:10.1609/AAAI.V38I17.29946

- [58] Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. 2025. MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. CoRR abs/2506.15841 (2025). arXiv:2506.15841 doi:10.48550/ARXIV.2506.15841
- [59] Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang, and Michael Bendersky. 2023. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, Hsin-Hsi Chen, Wei-Jou (Edward) Duh, HenHsen Huang, Makoto P. Kato, Josiane Mothe, and Barbara Poblete (Eds.). ACM, 2308–2313. doi:10.1145/3539618.3592047
- [60] Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon.

2025. Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning. CoRR abs/2503.06034 (2025). arXiv:2503.06034 doi:10. 48550/ARXIV.2503.06034

#### A Experimental Details

Implementation details. We use Qwen3-4B-Thinking [37] as the base model and train it for three rounds following the training method introduced in Section 3.4. During training, we adopt a 128k long input window, a 16k formal output window, and a 4k buffer window. The batch size is set to 32, and the number of GRPO [32] sampling times 𝑛 for each sample is set to 6. The RL training experiments of the memory proxy are conducted on an 8-GPU H200 machine. The working LLM calls the external Qwen3-30B-A3BInstruct API.

For training data preparation, we sample training data from the training sets of multiple memory-related benchmarks and restructure them to fit our method. We split the sessions and label each with a session ID following the input format specified in Section 3.2. For excessively long memories (exceeding 128K tokens), we sample memory content while ensuring that all relevant memories are included. More details are in Appendix C.

Session construction. When a benchmark provides session boundaries, we keep the original boundaries. For deep research trajectories, one search step or reasoning step is treated as one session. This session-level design is cheaper than turn-level retrieval and keeps the full text inside each selected session. Its limitation is that noisy sessions may still contain irrelevant turns; turn-level or hierarchical retrieval is left as future work.

Coarse filtering. We use BGE-M3 for the first-stage filter only when the full history is longer than the 128K proxy window. We embed the current query and each session, rank sessions by cosine similarity, and keep top sessions until the token budget is reached. We do not use a fixed similarity threshold. The full text of each kept session is then passed to the proxy.

Top-𝑘 setting. The proxy outputs a fixed top-10 ranked list. We use this fixed list size to keep the comparison stable across datasets and baselines. The working LLM receives the same top-𝑘 budget under each method. We also reserve a top-𝑘 sensitivity table to report whether smaller or larger 𝑘 changes the cost-performance trade-off. For fair comparison across datasets, the token cost in this table counts only the memory/session tokens introduced by top-𝑘 selection and excludes the current query and fixed task prompt.

Cross-working-LLM setting. The memory proxy is optimized with task feedback from Qwen3-30B-A3B-Instruct. At test time, the retrieved memory is evaluated with both Qwen3-30B-A3B-Instruct and DeepSeek V3.2 as working LLMs. This setting tests whether the proxy learns a retrieval policy that transfers across working LLMs.

Training cost accounting. We report training cost using GPU node hours, working LLM calls, input/output tokens, and API cost. All threetraining rounds arecompleted on one 8×H200 node. The reward model calls Qwen3-30B-A3B-Instruct through the SiliconFlow API, charged at RMB 0.70 per million input tokens and RMB 2.80 per million output tokens. The API token counts in Table 7 are billinglevel estimates reconstructed from this price schedule and the final provider bill. For top-10 reward evaluation, dense evaluation would require the no-memory baseline plus every rank cutoff, i.e., 11 working-LLM calls. Fibonacci sampling evaluates {0, 1, 2, 3, 5, 8, 10} instead, reducing the reward-evaluation calls to 7 per rollout.

##### Table 5: Train/test split statistics. Train and test sets are strictly disjoint.

Dataset Scale Train Test LoCoMo 32K 1,136 400 LongMemEval 128K 350 150 LongMemEval 1M 350 150 PersonaMem 32K 0 400 PersonaMem 128K 2,303 400 PerM-V2 128K 3,000 400 ZH4O 128K 668 400 HotpotQA 128K 3,000 400 WebWalker 128K 3,000 400 WebDancer 128K 3,000 400 Total – 15,807 3,500

##### Table 6: Coarse filtering statistics.

Dataset Recall Ret. Tok.

LoCoMo-32K 100.00 22.2K LongMemEval-128K 100.00 105.0K LongMemEval-1M 94.88 128.0K PersonaMem-32K 100.00 24.3K PersonaMem-128K 100.00 117.1K PerM-V2-128K 96.50 128.0K ZH4O-128K 100.00 20.0K

##### Table 7: Training cost accounting.

###### Item Value

GPU hardware 1 node, 8×H200 GPU node hours ≈96 node-hours Training rounds 3 Working LLM calls ≈200K Input tokens ≈250M Output tokens ≈42M API cost ≈RMB 292 ($40) Fibonacci sampling call saving 7 vs. 11 calls per rollout (36.4%)

Direct-rank ablation. To test the need for reasoning before retrieval, we add a direct-rank variant. The standard MemSifter proxy follows a Think-and-Rank format, first emitting “<think>” and then “<ranking>”. The direct-rank variant removes the “<think>” output requirement and asks the proxy to output only “<ranking>”. All other settings are kept the same, including the base model, train/test split, top-𝑘 budget, working LLM, and evaluation metric. The reporting set covers LoCoMo-32K, LongMemEval-1M, and WebDancer-128K, with F1, output tokens, and latency.

#### B Case Study

To provide a qualitative perspective on MemSifter’s effectiveness, we present case studies on LoCoMo (Figure 5) and WebDancer (Figure 6). These examples show the memory proxy’s reasoning process and the ranked sessions it selects for the working LLM.

##### Table 8: Inference-time top-𝑘 sensitivity results. Tok. denotes the estimated average memory/session tokens introduced by top-𝑘 selection, excluding the current query and fixed task prompt.

Top-𝑘 LoCoMo 32K LongMemEval 1M WebDancer 128K F1 Tok. F1 Tok. F1 Tok.

3 40.02 2.4K 30.38 6.5K 36.64 6.9K 5 41.17 4.0K 31.81 10.9K 37.10 11.6K 10 41.79 8.0K 33.32 21.8K 38.21 23.1K 15 41.46 12.0K 32.95 32.7K 37.87 34.7K

This constructed environment serves as astress testfor retrievalaugmented generation. It evaluates whether a method can effectively sift through massive volumes of noisy search logs and reasoning chains to extract the specific needle-in-a-haystack evidence required for a correct final answer.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

##### Table 9: PersonaMem multiple-choice accuracy. Accuracy is computed by normalizing each generated answer to the predicted option before exact-match scoring. The best and second best of each setting are in bold and underlined.

DeepSeek V3.2 Qwen3-30B

Method

32K 128K 32K 128K

BGE-M3 31.50 23.75 36.00 30.50 GemmaEmb 32.75 23.50 31.75 27.75 Nemori 32.75 25.25 34.25 29.75 Mem0 35.00 26.75 35.75 28.00 MemAgent 29.50 24.75 36.75 27.75 Mem-𝛼 37.25 25.75 34.00 28.75 HippoRAG 37.00 24.75 33.25 27.25 A-MEM 33.00 22.75 35.25 27.25 Rearank 34.50 23.75 31.25 30.75 ReasonRank 33.50 27.25 34.00 26.00 Long-context LLM 33.50 25.50 34.00 27.25 MemSifter 38.50 28.00 40.50 31.25

#### C Deep Research Dataset Construction

To rigorously evaluate the capability of memory systems in handlingcomplex,multi-stepinformation seeking tasks, we constructed a specialized “Deep Research” benchmark. The core search trajectories and reasoning traces are sampled from the MiroVerse dataset 2.

However, to differentiate the performance of advanced memory methods from standard baselines, we significantly increased the difficulty of the original data through two key modifications:

##### (1) Noise Injection via Semantic Distractors: We augmented

the original search trajectories by injecting approximate search results. These distractors are semantically related to the query but contain factually irrelevant or misleading details. This design forces the memory model to perform precise discrimination based on logical consistency rather than simple keyword matching.

###### (2) Context Extension: We concatenated multiple search it-

erations and intermediate reasoning steps to create significantly longer context windows.

2https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1

Question: Where am I planning to stay for my birthday trip to Hawaii? Answer: Oahu. Reasoning: To determine the most relevant sessions based on the two inputs provided (the Historical Interaction Information and the current chat context), I need to analyze both of them. The current chat context that was specified is: "Where am I planning to stay for my birthday trip to Hawaii?" Now, let’s look at the Historical Interaction Information to find the most relevant sessions related to Hawaii. From the History:

- - <session0> talks about planning a birthday trip to Hawaii with the user asking about hiking trails on Kauai
- - <session27> discusses a birthday trip to Hawaii in October, with the user asking about snorkeling spots on the island
- - <session24> mentions planning a family vacation to Hawaii with information about resorts that can accommodate a large group
- - <session27> again talks about the birthday trip with more details about snorkeling spots
- - <session0> has more details about hiking trails on Kauai
- - <session4> talks about a family reunion (with the user mentioning their grandfather’s 80th birthday celebration)
- - <session6> discusses planning a family reunion for next summer, with the user mentioning their nephew’s high school graduation
- - <session4> mentions the grandfather’s favorite dishes
- - <session2> has more details about the family reunion
- - <session1> talks about the location of the family reunion (the cabin in the mountains) I could also consider more:
- - <session8> (more details about the family reunion)
- - <session9> (more information about the activities)
- - <session10> (more details about the trip)
- - <session11> (more information about the location)
- - <session12> (more details about the activities)
- - <session13> (more information about the culture)
- - <session14> (more details about the trip)
- - <session15> (more information about the location)
- - <session16> (more details about the activities)
- - <session17> (more information about the culture) Based on the topic consistency (all related to Hawaii), user need continuity (the user mentions planning a birthday trip), and detail overlap (the sessions have information that matches the current context), the most relevant sessions are: · · · (Truncated Verification Process) · · · I think I can rank these in descending order of relevance as follows:

- 1. <session0> (high relevance - talks about hiking trails, with the user mentioning the birthday trip)
- 2. <session27> (more details about the trip, with the user mentioning snorkeling spots)
- 3. <session24> (more information about the family vacation with details about the accommodations)
- 4. <session4> (more details about the family reunion with the user mentioning the grandfather’s birthday)
- 5. <session6> (more information about the family reunion with the user mentioning the nephew’s graduation)
- 6. <session1> (more details about the location and activities)
- 7. <session2> (more information about the food)
- 8. <session3> (more details about the history)
- 9. <session5> (more information about the trip)
- 10. <session7> (more details about the culture) This ranking is based on the topic consistency (all related to Hawaii), user need continuity (the user mentions planning a birthday trip), and detail overlap (the sessions have information that matches the current chat context). </think> <ranking>0,27,24,4,6,1,2,3,5,7</ranking>

##### Figure 5: Case study for the reasoning trajectory on LoCoMo.

Question: Which actor portrayed the father of the main character in the adaptations of a popular book series by J.K. Rowling, known for playing the role of the main character’s father in a widely recognized film series? Answer: Adrian Rawlins. Reasoning: Let me analyze the two inputs provided: the Historical Interaction Information and the current chat context, to rank the most relevant sessions. The current chat context given is: "Which actor portrayed the father of the main character in the adaptations of a popular book series by J.K. Rowling, known for playing the role of the main character’s father in a widely recognized film series?" Now, let’s look at the Historical Interaction Information to find the most relevant sessions related to J.K. Rowling’s work. From the list, I can see:

- - <session63>: Which actor portrayed the father of the main character in the adaptations of a popular book series by J.K. Rowling, known for playing the role of the main character’s father in a widely recognized film series?
- - <session9>: Identifying the Son of Paul Simon (with information about Paul Simon, the performer of "Graceland", and his sons)
- - <session33>: Who Played Harry Potter’s Father in the Film Adaptations (with information about Adrian Rawlins, who played James Potter, Harry Potter’s father)
- - <session21>: Final Response: Country of Origin for the Album ’69 Love Songs’ (with information about the album and its origin)
- - <session2>: About the TV show ’Gotham’ (with information about its production and location)
- - <session0>: About a High Court jurist (with information about their work and background)
- - <session1>: About the economic outlook of India (with data from Deloitte)
- - <session2> (again, about the TV show ’Gotham’)
- - <session3>: About the financial regulations of a university (with information about the implementation date)
- - <session4>: About the ownership structure of a company (with details about its structure)
- - <session5>: About the history of a region (with information about its division and the population)
- - <session6>: About a sports celebrity dinner (with details about the event)
- - <session7>: About a lyricist born in the Palatinate region (with information about their work and background)
- - <session8>: About the UNESCO World Heritage Sites (with information about the sites and their status)
- - <session9>: More information about Paul Simon
- - <session10>: About the Diocese of Bath and Wells (with details about its structure and the bishop)
- - <session11>: About the percentage composition of a compound
- - <session12>: About the seat of Brown County, Wisconsin
- - <session13>: About the Court of the Lord Lyon (with information about a ruling)
- - <session14>: About Oriel College Oxford (with details about its history and the alumni)
- - <session15>: About the Visalia Fox Theatre (with information about its history and connection to a musician)
- - <session16>: About the ethical mining of rose quartz (with details about the industry)
- - <session17>: About a medical trial (with information about the research and methodology)
- - <session18>: About the Hudson Bay drainage basin (with details about its geography and history)
- - <session20>: About an infomercial-focused television channel in Australia
- - <session21> (again, about the album ’69 Love Songs’)
- - <session22>: About the IETF 122 event (with details about the conference)
- - <session23>: About the van der Waals equation (with mathematical information)
- - <session24>: About the South Atlantic (with information about its geography and history)
- - <session25>: About the Indonesian Film Festival (with details about its history and significance) · · · (Truncated Verification Process) · · · Based on the topic consistency (the most relevant sessions should match the current chat topic) and user need continuity (they are related to the user’s current need), the most relevant sessions for the current chat context are: · · · (Truncated Verification Process) · · · After reviewing, the most relevant sessions (highest relevance) are:

- 1. <session63> (the user mentioned the topic in the current chat context)
- 2. <session33> (more details about the same topic)
- 3. <session9> (more information about the author and his family)
- 4. <session21> (more details about the work)
- 5. <session2> (more information about the production)
- 6. <session0> (more details about the person)
- 7. <session1> (more information about the work)
- 8. <session3> (more details about the implementation)
- 9. <session4> (more information about the company)
- 10. <session5> (more details about the history) This ranking is based on topic consistency (the sessions match the current chat topic) and user need continuity (they are related to the user’s current need for information about the actor who portrayed the father of the main character in the adaptations of J.K. Rowling’s book series). </think> <ranking>63,33,9,21,2,0,1,3,4,5</ranking>

##### Figure 6: Case study for the reasoning trajectory on WebDancer.

