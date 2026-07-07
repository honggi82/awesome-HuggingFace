## DLLM-Searcher: Adapting Diffusion Large Language Models for Search Agents

Jiahao Zhao∗ Shaoxuan Xu∗

Zhongxiang Sun∗†

Fengqi Zhu Jingyang Ou

Renmin University of China Beijing, China sunzhongxiang@ruc.edu.cn

Renmin University of China Beijing, China zhaojiahao2202@ruc.edu.cn

Renmin University of China Beijing, China fengqizhu@ruc.edu.cn

Jun Xu‡ Xiao Zhang

Yuling Shi

Chongxuan Li

Shanghai Jiao Tong University Shanghai, China yuling.shi@sjtu.edu.cn

Renmin University of China Beijing, China chongxuanli@ruc.edu.cn

# arXiv:2602.07035v1[cs.AI]3Feb2026

Renmin University of China Beijing, China junxu@ruc.edu.cn

### Abstract

###### LLM-Searcher w/ ReAct

Think Token Tool Call Token

[Figure 1]

[Figure 2]

[Figure 3]

Recently, Diffusion Large Language Models (dLLMs) have demonstrated unique efficiency advantages, enabled by their inherently parallel decoding mechanism and flexible generation paradigm. Meanwhile, despite the rapid advancement of Search Agents, their practical deployment is constrained by a fundamental limitation, termed as 1) Latency Challenge: the serial execution of multiround reasoning, tool calling, and tool response waiting under the ReAct agent paradigm induces severe end-to-end latency. Intuitively, dLLMs can leverage their distinctive strengths to optimize the operational efficiency of agents under the ReAct agent paradigm. Practically, existing dLLM backbones face the 2) Agent Ability Challenge. That is, existing dLLMs exhibit remarkably weak reasoning and tool-calling capabilities, preventing these advantages from being effectively realized in practice. In this paper, we propose DLLM-Searcher, an optimization framework for dLLM-based Search Agents. To solve the Agent Ability Challenge, we design a two-stage post-training pipeline encompassing Agentic Supervised Fine-Tuning (Agentic SFT) and Agentic Variance-Reduced Preference Optimization (Agentic VRPO), which enhances the backbone dLLM’s information seeking and reasoning capabilities. To mitigate the Latency Challenge, we leverage the flexible generation mechanism of dLLMs and propose a novel agent paradigm termed ParallelReasoning and Acting (P-ReAct). P-ReAct guides the model to prioritize decoding tool_call instructions, thereby allowing the model to keep thinking while waiting for the tool’s return. Experimental results demonstrate that DLLM-Searcher achieves performance comparable to mainstream LLM-based search agents and P-ReAct delivers approximately 15% inference acceleration. Our code is available at https://anonymous.4open.science/r/DLLM-Searcher-553C

[Figure 4]

. . . Query × N

[Figure 5]

[Figure 6]

Success

Think Tool Call

|<tool_response> … </tool_response><br><br>[Figure 7]<br><br>Generating . . . Waiting . . .<br><br>[Figure 8]<br><br>[Figure 9]<br><br><think> …</think> <think> …</think> <tool_call> …</tool_call ><br><br>||Long Latency|
|---|
<br><br>Gen Wait|
|---|---|

###### DLLM w/ ReAct

Format Error!

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

. . .× N

###### Retry

Query Fail

Think Tool Call

<think> …</think> <tool_call> …</tool>

[Figure 14]

###### DLLM-Searcher w/ P-ReAct

[Figure 15]

Masked Token

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

. . . × N

Tool Call Parallel !

Query

Success

[Figure 20]

Think

<tool_response> … </tool_response>

| | |
|---|---|
| | |

<tool_call> …</tool_call > No

<mask> <mask>

<think> …</think>

Waiting

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 21]

Gen Wait

Generating . . .

[Figure 22]

Short Latency

Figure 1: LLM-based Search Agent (ReAct) vs. dLLM-based Search Agent (ReAct and P-ReAct). Top & Middle: Standard ReAct paradigm suffers from high latency due to serial thinking and tool-calling regions generation and tool response waiting, while vanilla dLLMs fail due to tool-calling format errors. Bottom: P-ReAct prioritize tool-calling generation, enabling the model to keep thinking during tool execution.

∗Both authors contributed equally to this research. †Project Leader ‡Corresponding author.

### 1 Introduction

In recent years, Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to traditional Autoregressive Models (ARMs) [22, 41, 43]. While ARMs are fundamentally constrained by a sequential “left-to-right” next-token prediction process, dLLMs leverage a non-causal diffusion mechanism that enables two advantages: parallel decoding mechanism and flexible generation paradigm [1, 7, 10, 22, 31, 41].

This work is licensed under a Creative Commons Attribution 4.0 International License. Conference’17, Washington, DC, USA

© 2026 Copyright held by the owner/author(s). ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

Concurrently, the integration of information retrieval with Large Language Models (LLMs) has led to the emergence of Search Agents, which enable LLMs to autonomously invoke tools to enhance generation quality [15, 28–30]. Empowered by agentic post-training, these agents predominantly operate under the Reasoning and Acting (ReAct) agent paradigm [40]. Under this paradigm, an agent first generates a think region to devise a search plan, followed by a tool_call region to translate this plan into an API request and then halts generation to wait for external feedback. However, this serial execution creates a 1) Latency Challenge: the end-to-end response time is severely bottlenecked by the cumulative delays of think and tool_call generation and waiting for the tool response, as the model remains inactive during external tool execution.

Intuitively, dLLMs are ideal to mitigate this latency challenge. Beyond supporting parallel decoding for accelerated generation, their flexible generation paradigm offers significant potential for restructuring the ReAct execution flow. However, a significant gap exists between the theory and practice. As illustrated in Figure 1, vanilla dLLMs frequently fail to adhere to specific tool-calling formats. Furthermore, their performance generally trails behind that of ARMs, particularly in agentic scenarios that demand robust reasoning and strict format compliance. These deficiencies constitute a critical 2) Agent Ability Challenge, which hinders the practical deployment of dLLMs as search agent backbones.

In this paper, we propose DLLM-Searcher, an optimization framework that effectively enhances the information seeking and reasoning ability of dLLMs and improves their efficiency in agent scenarios. To address 2) Agent Ability Challenge, we design a twostage post-training strategy, which is specifically tailored for improving dLLMs’ agentic capability. To further exploit the strengths of dLLMs and boost the efficiency of dLLM-based search agents to deal with 1) Latency Challenge, we propose the Parallel-Reasoning and Acting (P-ReAct), a novel agent paradigm that parallelizes the thinking and waiting phases.

Training Process: Specifically, our post-training pipeline is grounded in Agentic ELBO, a loss estimation method tailored for dLLM agents, and proceeds in two stages. First, we conduct Agentic Supervised Fine-Tuning (Agentic SFT). This stage uses trajectories derived from a set of multi-hop questions to enable the model to acquire foundational information-seeking capabilities adapted to large block sizes. Subsequently, we employ Agentic VarianceReduced Preference Optimization (Agentic VRPO). By utilizing data filtered from post-SFT model rollouts, this stage further refines the model’s reasoning and retrieval performance.

Inference process: The P-ReAct agent paradigm includes two critical components: Token Pre-filling and Confidence Biasing. We pre-fill <tool_call> and </tool_call> in the latter part of the block, and add a bias to the confidence scores of the positions between these two special tokens during the decoding process, guiding the model to prioritize decoding the content within this region.

We evaluate DLLM-Searcher along with existing Retrieval Augmented Generation (RAG) frameworks and search agents over four multi-hop benchmarks [9, 25, 34, 39]. Experimental results show that the information seeking and reasoning capability of DLLMSearcher is comparable to mainstream LLM-based search agents such as R1Searcher [28]. Moreover, it ensures with a nearly 100%

success rate that the tool-calling part is decoded first in every PReAct iteration process.

In summary, the contributions of our paper are as follows:

- (1) DLLMs Agent Post-training. We develop a post-training pipeline for dLLMs, comprising Agentic SFT and Agentic VRPO, which can enhance dLLMs’ agent capability.
- (2) DLLMs Agent Paradigm. We propose P-ReAct, a novel, training-free paradigm adapted for dLLMs that guides the model to prioritize decoding high-quality tool calls, enabling parallel execution.
- (3) DLLM-Searcher Performance. DLLM-Searcher achieves performance comparable to LLM-based search agents and realizes approximately 15% acceleration compared to the ReAct paradigm.

2 Related Work

- 2.1 Diffusion Large Language Models

Diffusion Language Models. Inspired by discrete diffusion models [2, 5, 20, 21, 42], dLLMs have emerged as a promising alternative to ARMs. LLaDA [22], an 8B diffusion language model trained from scratch, achieves performance competitive with LLaMA3-8B. Dream7B [41] introduces a comprehensive training framework that leverages AR-based LLM initialization and context-adaptive noise scheduling to scale diffusion language models. More recently, industrial efforts such as Gemini Diffusion [7], Mercury [10], and Seed-Diffusion [31] have further scaled dLLMs and demonstrated their potential for efficient inference.

Block Diffusion Language Models. Furthermore, hybrid architectures such as Block Diffusion Language Models (BDLMs) [1, 3, 6, 37] have become a significant research focus. BDLMs employ an attention mechanisms combining intra-block bidirection with interblock causal. This architecture natively supports KV Cache and variable-length text generation and keep the non-autoregressive generation ability within each block, allowing parallel decoding in arbitrary orders. SDAR [6], a series of BDLMs ranging from 1.7B to 8B parameters, has demonstrated general-purpose capabilities comparable to the latest open-source ARMs.

Despite these architectural advantages and competitive performance on tasks such as mathematics, existing dLLMs still lag behind ARMs in complex reasoning and agentic tasks. Weak reasoning capabilities and poor instruction-following ability prevent dLLMs from serving as agent backbone models. Our work enhances the agentic capabilities of dLLMs through a two-stage post-training pipeline, while leveraging their flexible generation mechanism to accelerate agentic inference.

- 2.2 Search Agent

Recent advances in search agents aim to integrate web search tool calling with the reasoning process of LLMs, enabling models to autonomously retrieve external knowledge after thinking. This paradigm significantly mitigates hallucination issues and enhances generation quality by grounding responses in retrieved evidence [8, 27]. To further strengthen the synergy between reasoning and tool calling through training, researchers have explored diverse post-training strategies. R1-Searcher [28], Search-R1 [11] employ a two-stage post-training pipeline consisting of Supervised

Fine-Tuning (SFT) followed by Reinforcement Learning (RL) on open-source datasets, demonstrating substantial improvements in both reasoning and search capabilities. WebSailor [15] synthesize more challenging questions to push the boundaries of model search and reasoning abilities. MiroThinker [33] achieves superior performance by scaling the number of search iterations.

However, all these agents adopt the ReAct paradigm [40], where reasoning, tool calling, and waiting for tool responses are executed serially. This sequential execution pattern forces users to endure prolonged waiting times, making latency a critical bottleneck for practical Search Agent deployment. DLLM-Searcher addresses this challenge by breaking the serial mechanism of ReAct, leveraging the flexible generation paradigm of dLLMs to enable parallel reasoning and action execution.

3 Preliminary

- 3.1 Diffusion Large Language Models

Formally, dLLMs model the data distribution through a forwardreverse framework. In the forward diffusion process, as the time step 𝑡 advances from 0 to 1, the clean input sequence 𝑦 is progressively corrupted by replacing tokens with a special mask token [M] according to a transition probability 𝑞𝑡. Consequently, given the time step 𝑡 sampled uniformly from the interval [0, 1], 𝑦 = (𝑦1, . . .,𝑦𝐿) denote a clean input sequence of length 𝐿, and the conditioning prompt 𝑥, the transition probability 𝑞𝑡 is formulated as:

𝑞𝑡 (𝑦𝑡 | 𝑦,𝑥) =

𝐿

𝑖=1

𝑞𝑡 (𝑦𝑡𝑖 | 𝑦𝑖,𝑥), (1)

𝑞𝑡 (𝑦𝑡𝑖 | 𝑦𝑖,𝑥) =

1 − 𝑡, 𝑦𝑡𝑖 = 𝑦𝑖, 𝑡, 𝑦𝑡𝑖 = [M].

(2)

In the reverse process, the model predicts the original values of

the masked tokens to compute the reverse probability𝑝𝜃 (·|𝑦𝑡,𝑥). To learn this distribution effectively, dLLMs adopts the Evidence Lower

Bound (ELBO) L𝜃 (𝑦 | 𝑥) as a surrogate objective to approximate the log conditional distribution log𝜋𝜃 (𝑦|𝑥) [22–24]:

L𝜃 (𝑦 | 𝑥) ≜ E𝑡∼U[0,1],𝑦𝑡∼𝑞𝑡 (𝑦𝑡 |𝑦,𝑥) (3)

1 𝑡

∑︁𝐿

𝑖=1

1[𝑦𝑡𝑖 = [M]] log𝑝𝜃 (𝑦𝑖 | 𝑦𝑡,𝑥) ≤ log𝜋𝜃 (𝑦 | 𝑥).

For BDLMs, the input 𝑦 is partitioned into 𝐾 continuous blocks [𝑦1, . . .,𝑦𝐾], each of length 𝐵. The ELBO is defined as:

L𝜃𝑏𝑙𝑜𝑐𝑘(𝑦 | 𝑥) ≜ E𝑡∼U[0,1],𝑦𝑡∼𝑞𝑡 (𝑦𝑡 |𝑦,𝑥) (4)

1 𝑡

∑︁𝐾

𝑘=1

∑︁𝐵

𝑖=1

1 𝑦𝑡𝑘,𝑖 = [M] log𝑝𝜃 (𝑦𝑘,𝑖 | 𝑦𝑡𝑘,𝑦<𝑘,𝑥) .

- 3.2 Search Agents with ReAct

In this section, we formally define the trajectory of a search agent interacting with the environment under the ReAct framework. Search Agents typically adopt ReAct as the agent framework. Let 𝑓𝜃 denote the agent LLM parameterized by 𝜃. Upon receiving a query 𝑄 from the user, the agent follows the system prompt 𝑆 and performs several iterations of Thought-Action-Observation.

We define the initial context as H0 = (𝑆,𝑄), which consists of the system prompt and the user query. At the 𝑛-th iteration (𝑛 ≥ 1), let 𝑇𝑛, 𝐴𝑛, and 𝑂𝑛 denote the thought, action, and observation, respectively. Based on the existing context H𝑛−1 from previous iterations, the agent generates a thought𝑇𝑛 and executes a parsable action 𝐴𝑛:

(𝑇𝑛,𝐴𝑛) = 𝑓𝜃 (H𝑛−1),

then waits for the environment to return an observation 𝑂𝑛. The context is updated as H𝑛 = (H𝑛−1,𝑇𝑛,𝐴𝑛,𝑂𝑛). In search scenarios, the action space consists of generating a final answer and calling the search tool with agent generated queries. The iteration terminates when the agent selects final answer as the action.

Assuming the process terminates after 𝑁 iterations, the complete trajectory can be defined as:

H𝑁 = (𝑆,𝑄,𝑇1,𝐴1,𝑂1, . . .,𝑇𝑖,𝐴𝑖,𝑂𝑖, . . .,𝑇𝑁,𝐴𝑁).

4 Our Approach: DLLM-Searcher

- 4.1 Overview

As illustrated in Figure 2, DLLM-Searcher consists of (i) a two-stage post-training pipeline and (ii) the P-ReAct agent paradigm.

Two-stage post-training pipeline. Our experiments Table 1 show that dLLMs are weak in both multi-step reasoning and strict toolcall format following, which motivates Agentic post-training. Agentic SFT (§ 4.2). We construct training data using trajectories generated by a stronger teacher model, in the form of (𝑄, Hteacher). This stage improves the model’s tool-call format following ability and helps it acquire initial capabilities to combine information retrieval with reasoning under large-block generation. Agentic VRPO (§ 4.3). Starting from the SFT model, we roll out the trajectories and then filter them into winner/loser pairs (𝑄, H𝑤, H𝑙) based on correctness. We then apply VRPO to further align the model toward correct trajectories, strengthening robust information-seeking behavior.

P-ReAct agent paradigm. dLLMs can generate tokens in an arbitrary order within a block, but it’s difficult to precisely control the generation order. Through extensive exploration, we develop a training-free strategy: we pre-fill the special boundary tokens <tool_call> and </tool_call> in the first step, and during subsequent decoding we apply a positional confidence bias to the token positions between these two boundaries. P-ReAct (§ 4.4) encourages the model to prioritize decoding the tool-call, effectively ensuring that tool-call instructions are generated ahead of the thinking process with near-perfect controllability.

- 4.2 Agentic SFT

We evaluate the agent capabilities of existing dLLM backbones and find that: dLLMs, particularly the BDLMs adopted in our work exhibit certain general-purpose abilities, they still fall short of the requirements in the Search Agent setting. Especially, they lack multi-step reasoning and tool-calling abilities. Therefore, we perform Agentic SFT to improve these capabilities. Furthermore, Search Agent trajectories typically contain external web content returned by search engines, whereas we want the model to learn only the think and tool_call regions. Meanwhile, dLLMs are commonly

#### Train Process

###### Block Attention

###### Agentic Noising

Prompt Think Tool call / Answer No Attention Attention Tool Response Mask

0 1 2 3 4 5 2 3 4 5

[tool resp | think]

[tool resp | noisy]

[Figure 23]

- 0

- 1

- 2

- 3

- 4

- 5

- 0
- 1
- 2
- 3
- 4
- 5

[tool resp]

[tool resp]

[noisy | mask]

[tool call | tool resp]

- 2

- 3

- 4

- 5

| | | | | |
|---|---|---|---|---|

- 2
- 3
- 4
- 5

| |
|---|

[think | tool call]

[noisy]

Agentic Random Mask

###### Compute ELBO

[Figure 24]

[Figure 25]

[Figure 26]

Agentic SFT

Agentic VRPO

| |
|---|
| |
| |

| |by|
|---|---|
| | |

[Figure 27]

Judged

[Figure 28]

[Figure 29]

Filter by our rules

[Figure 30]

ReAct

[Figure 31]

Our Rollout

#### Inference Process using P-ReAct

[Figure 32]

[Figure 33]

[Figure 34]

Confidence

Confidence

[Figure 35]

[Figure 36]

Searching ...

Tool Call decoding process DONE！

[Figure 37]

[Figure 38]

Add Bias

Parsing the arguments ... Processing the docs ...

[Figure 39]

<think> I need to search ...

Position

Position

| | | | | |
|---|---|---|---|---|

| | | |
|---|---|---|

| | | | | |
|---|---|---|---|---|

Prompt

Prompt Tool Call Prompt Think Tool Call Tool Resp

<tool_call>

</tool_call>

- Figure 2: DLLM-Searcher includes training process and inference process. In training, both Agentic SFT and Agentic VRPO use

Block Attention and Agentic Noising to compute the Agentic ELBO, which serves to estimate log𝜋𝜃 (𝑦 | 𝑥). In inference, we employ the P-ReAct agent paradigm . We pre-fill special boundary tokens and apply an additional confidence bias to encourage the model to decode the tool_call region with priority.

optimized by maximizing the ELBO. To reconcile these characteristics, we propose Agentic Noising process and Agentic ELBO tailored for dLLM-based agent training.

- 4.2.1 Data Construction. Givenaquery𝑄,we usehigh-performance

models to generate a teacher trajectory Hteacher. We then apply a filter to retain only trajectories with correct final answers, clear and complete reasoning steps, and strictly valid tool call formats. The remaining query trajectory pairs (𝑄, Hteacher) are used as training data.

- 4.2.2 Block Attention and Agentic Nosing. As illustrated in Figure 2, the Block Attention used to train BDLMs adopts bidirectional attention within each block and causal attention across blocks. During training, we concatenate a noised trajectory after

the clean one, forming an input of the form [𝑄, Hteacher, Hnoisy], which aims to perform calculations for all noisy blocks conditioned on their corresponding clean blocks in a single forward process

under block attention. Since we want the model to learn only the think and tool_call parts, we inject noise only into these components. Moreover, due to intra-block bidirectional attention, without additional intervention the model could access information from tool_response tokens that appear later in the same block after the generation tokens, leading to a train inference mismatch. Therefore, in such cases we must fully mask the tool_response tokens. We adopt the forward diffusion process 𝑞ˆ𝑡 described above to noise Hteacher into Hnoisy, with the detailed procedure given in Algorithm 1.

4.2.3 Agentic ELBO. Correspondingly, regarding the optimization objective, we aim for the model to focus on learning the think and tool_call regions. Therefore, we adapt the Eq. 4 to Agentic ELBO

Algorithm 1 Agentic Noising Process 𝑞ˆ𝑡 Input: sequence of tokens 𝑦, block size 𝐵, diffusion timestep 𝑡, set

of tool response tokens [R], mask token [M] Result: noised sequence 𝑦𝑡 for training

- 1: Initialize 𝑦𝑡 ← 𝑦
- 2: Partition 𝑦 into blocks {𝑦1,𝑦2, . . .,𝑦𝐾} of size 𝐿
- 3: for each block 𝑦𝑘 do
- 4: tool resp tokens’ indices: Iresp ← {𝑖 ∈ 𝑦𝑘 | 𝑦(𝑖) ∈ [R]}
- 5: gen tokens’ indices: Iother ← {𝑗 ∈ 𝑦𝑘 | 𝑦(𝑗) ∉ [R]}
- 6: if 𝐼resp = ∅ then
- 7: ∀𝑗 ∈ 𝑦𝑘 : 𝑦𝑡(𝑗) ∼ 𝑞𝑡 (𝑦𝑡(𝑗) | 𝑦(𝑗),𝑥) ⊲ 1. Pure Gen Block
- 8: else if 𝐼other = ∅ then
- 9: Skip ⊲ 2. Pure Response Block
- 10: else
- 11: 𝑖𝑑𝑥𝑅 = min(Iresp)
- 12: 𝑖𝑑𝑥𝑂 = min(Iother)
- 13: if 𝑖𝑑𝑥𝑅 < 𝑖𝑑𝑥𝑂 then
- 14: ∀𝑖 ∈ Iresp : 𝑦𝑡(𝑖) ← [M]
- 15: ∀𝑗 ∈ Iother : 𝑦𝑡(𝑗) ∼ 𝑞𝑡 (𝑦𝑡(𝑗) | 𝑦(𝑗),𝑥)
- 16: ⊲ 3. Leakage Risk
- 17: else
- 18: ∀𝑖 ∈ Iresp : 𝑦𝑡(𝑖) ← 𝑦(𝑖)
- 19: ∀𝑗 ∈ Iother : 𝑦𝑡(𝑗) ∼ 𝑞𝑡 (𝑦𝑡(𝑗) | 𝑦(𝑗),𝑥)
- 20: ⊲ 4. Observation Context
- 21: end if
- 22: end if
- 23: end for
- 24: return 𝑦𝑡

Lˆ𝜃𝑏𝑙𝑜𝑐𝑘 as follows: Lˆ𝑏𝑙𝑜𝑐𝑘𝜃 (𝑦 | 𝑥) ≜ E𝑡∼U[0,1],𝑦𝑡∼𝑞ˆ𝑡 (𝑦𝑡 |𝑦,𝑥) (5)

∑︁𝐾

###### ∑︁𝐵

1 𝑡

1 (𝑦𝑡𝑘,𝑖 = [M]) ∧ (𝑦𝑘,𝑖 ∉ [R]) log𝑝𝜃 (𝑦𝑘,𝑖 | 𝑦𝑡𝑘,𝑦<𝑘,𝑥) ,

𝑖=1

𝑘=1

where we compute the loss only for tokens that are currently masked and were not originally in the tool_response regions. Since our Agentic Noising may mask response tokens to prevent leakage, we exclude such positions from contributing to the loss.

- 4.2.4 Training. We set 𝑦 = Hteacher and 𝑥 = 𝑆 + 𝑄. In standard LLM training, the objective is typically the token-level negative log-likelihood −log𝜋𝜃 (𝑦 | 𝑥). In our setting, we use the proposed Agentic ELBO to approximate log𝜋𝜃 (𝑦 | 𝑥). Therefore, the final training loss is defined as the negative Agentic ELBO

L𝑆𝐹𝑇 = E(𝑥,𝑦)∼D − Lˆ𝜃𝑏𝑙𝑜𝑐𝑘(𝑦 | 𝑥) .

### 4.3 Agentic VRPO

Inspired by prior work on post-training dLLMs [43], VRPO can further improve model capability on top of SFT. Therefore, to further enhance the model’s reasoning and information retrieval abilities and to better adapt it to our P-ReAct agent paradigm, we introduce Agentic VRPO. Specifically, we roll out trajectories using the SFT model equipped with P-ReAct, and construct training data from these trajectories to train the model. During loss computation, we

use the proposed Agentic ELBO introduced above to estimate 𝜋𝜃 (𝑦 | 𝑥) for both the reference model and the policy model.

- 4.3.1 Data Construction. Given a query𝑄, we perform two rollouts using the SFT model with P-ReAct to obtain two trajectories. We then select pairs where both trajectories are clear and complete and strictly follow the tool-call format, but one yields a correct final answer H𝑤 while the other yields an incorrect one H𝑙. The resulting training instance is formed as (𝑄, H𝑤, H𝑙).
- 4.3.2 Training. ConsistentwiththeSFTphase,weset𝑦𝑤 = H𝑤,𝑦𝑙 = H𝑙,𝑥 = 𝑆 + 𝑄, employ Agentic Noising 𝑞ˆ to ensure that tool responses do not disturbed the learning process. Consequently, we substitute the standard term in VRPO with our proposed Agentic ELBO Lˆ𝜃𝑏𝑙𝑜𝑐𝑘(𝑦 | 𝑥). The final objective is formulated as:

LVRPO(𝜃) ≜ E(𝑥,𝑦𝑤,𝑦𝑙)∼D − log𝜎 𝛽 ΔL(𝑦𝑤|𝑥) − ΔL(𝑦𝑙|𝑥) ,

where ΔL(𝑦|𝑥) ≜ Lˆ𝜃𝑏𝑙𝑜𝑐𝑘(𝑦 | 𝑥) − Lˆ𝑏𝑙𝑜𝑐𝑘ref (𝑦 | 𝑥) represents the Agentic ELBO advantage of the policy over the reference model, 𝛽

is a hyperparameter that controls the deviation from the reference policy.

### 4.4 P-ReAct Agent Paradigm

The bidirectional attention mechanism in dLLMs allows them to access global context from tokens that have not yet been explicitly decoded [16]. This provides a robust foundation for dLLMs to generate high-quality tool_call instructions by leveraging information from the underlying reasoning trajectory even before the reasoning steps are fully decoded. Howerver, Our experiments reveal that without specific intervention, the generation order of dLLMs is stochastic and difficult to control. Notably, the latest BDLM backbone (SDAR) used in this work is finetuned from ARMs. It tends to degenerate into an autoregressive, left-to-right generation sequence within a block. To address this, we propose P-ReAct. We demonstrate that by pre-filling the two boundary special tokens for tool calls and applying a confidence bias to the span between them, we can prioritize the decoding of the tool_call region with nearly 100% probability. This enables the immediate parsing and dispatching of parameters to the search engine, while the model continues to generate the think component during the waiting period.

4.4.1 Standard dLLMs Decoding. We first formalize the standard inference process of dLLMs, utilizing the Low-confidence Remasking strategy. Given a prompt 𝑥, the model generates a response sequence 𝑦 of length 𝐿 over 𝑁 denoising steps. In general, 𝐿 = 𝑘𝑁, implying that 𝑘 new tokens are decoded at each step. Let 𝑦𝑛 denote the sequence state at step 𝑛, M𝑛 be the set of masked indices at step 𝑛, and V be the vocabulary.

Initialization: Conventionally, the process begins with a fully masked sequence:

], M0 = {1, . . .,𝐿}.

𝑦0 = [[M], . . ., [M]

𝐿

Denoising Step: At each step 𝑛, the dLLM 𝑓𝜃 predicts logits Z𝑛 = 𝑓𝜃 (𝑦𝑛,𝑥) ∈ R𝐿×|V|. Only for a masked position 𝑖 ∈ M𝑛, we derive

the probability distribution 𝑃𝜃 (𝑦𝑖 | 𝑦𝑛,𝑥) = Softmax(𝑧𝑛𝑖 ). We define the predicted token 𝑦˜𝑖 and its corresponding confidence score 𝐶𝑛𝑖 as follows:

𝑦˜𝑖 = argmax

𝑃𝜃 (𝑦𝑖 = 𝑤 | 𝑦𝑛,𝑥), 𝐶𝑛𝑖 = max 𝑤∈V

𝑃𝜃 (𝑦𝑖 = 𝑤 | 𝑦𝑛,𝑥).

𝑤∈V

The remasking strategy then selects a subset of positions with the highest confidence scores, unmasks them with their predicted tokens to form 𝑦𝑛+1, and updates M𝑛+1.

- 4.4.2 P-ReAct: Controlled Decoding Strategy. The standard process described above implies an uncontrolled generation order. PReAct enforces a "Tool-First" hierarchy via two key modifications: tool_call Token Pre-filling and Confidence Biasing.

1. Special Token Pre-filling. To constrain the search space, we inject structural priors into the initialization 𝑦0. Instead of a fully masked sequence, we pre-fill the boundary tokens for tool calls at designated positions. Let 𝑝𝑜𝑠𝑠 and 𝑝𝑜𝑠𝑒 denote the start and end indices for the tool span, respectively:



<tool_call> if 𝑖 = 𝑝𝑜𝑠𝑠, </tool_call> if 𝑖 = 𝑝𝑜𝑠𝑒, [M] otherwise.

𝑦ˆ0(𝑖) =

(6)

 

By anchoring these boundaries, we explicitly define a structural skeleton in the noise space, forcing the model to generate valid tool content within the bracketed span.

2. Confidence Biasing. To ensure the content enclosed by the anchors is decoded prior to the reasoning text, we adjust the confidence ranking step. Specifically, during the decoding iterations, we inject a positive bias 𝛼 into the confidence scores of tokens located within the tool_call region:

𝐶ˆ𝑛𝑖 =

𝐶𝑛𝑖 + 𝛼, if 𝑝𝑜𝑠𝑠 < 𝑖 < 𝑝𝑜𝑠𝑒, 𝐶𝑛𝑖 , otherwise.

(7)

Given that the standard remasking strategy preferentially unmasks tokens with higher confidence, this bias effectively raises the decoding priority of the tokens within the tool_call region, guaranteeing their generation in the earlier decoding steps.

### 5 Experiments

In this section, we empirically verify the effectiveness of DLLMSearcher, First, we conduct extensive comparisons between DLLMSearcher and (i) traditional RAG methods, (ii) LLM-based agents, and (iii) dLLM-based agents, including our backbone SDAR, to verify that DLLM-Searcher improves the model’s information-seeking and reasoning capabilities. Then, to further analyze the effectiveness of the two core components of DLLM-Searcher, we formulate and answer the following research questions:

###### RQ1: Effectiveness of the two-stage post-training pipeline.

How does the proposed two-stage post-training pipeline, comprising Agentic SFT and Agentic VRPO, systematically enhance the information-seeking and reasoning abilities of dLLMs?

###### RQ2: Inference efficiency brought by P-ReAct. How does

P-ReAct achieve inference acceleration while maintaining performance?

RQ3: The advantage of order-free generation. Is P-ReAct a capability unique to dLLMs? Can autoregressive LLMs generate the tool_call region first without sacrificing performance?

Finally, we present a case study of a single P-ReAct iteration to qualitatively illustrate the “thinking-while-waiting” behavior exhibited by DLLM-Searcher in practice.

### 5.1 Experimental Settings

- 5.1.1 Datasets. This paper focuses on leveraging DLLM-Searcher to address complex multi-step question-answering (QA) tasks. To this end, four benchmark datasets are utilized in the experiments: HotpotQA[39], 2WikiMultiHopQA[9], Musique[34], Bamboogle[25]. Following the standard experimental setup of traditional RAG and search agent [12, 13, 28–30, 32, 36], we sampled 500 examples from the development sets of HotpotQA, 2WikiMultiHopQA, and Musique as the test sets. For Bamboogle, which has only 125 examples in its test set and all of them are used in the experiments.

To construct high-quality training data for DLLM-Searcher’s Agentic SFT, we design a trajectory sampling, rollout, and filtering pipeline. Specifically, we randomly sampling 2048 queries from each of the training sets of HotpotQA, 2WikiMultiHopQA, and Musique. Considering that Doubao-Seed-1.8 (251228) [4] is a recently released model with public API access, which demonstrates state-of-the-art performance in comprehensive capabilities, especially in search-related tasks, we utilize Doubao-Seed-1.8 to perform trajectory rollout with only one rollout iteration performed. Subsequently, we employ this model as the LLM judger with the prompt provided in our codebase. After that, we filter out trajectories that pass the LLM judge evaluation, feature complete reasoning paths, and comply with the standard tool_call format, which are then used as training data for the Agentic SFT, resulting in a curated dataset of 3977 trajectories.

For the Agentic VRPO, we utilize the SFT model to perform two rounds of rollouts on the 8k Stage 2 training samples released by R1Searcher. We then filter for queries where one rollout yields a correct answer while the other produces an incorrect one, with both corresponding trajectories being complete and format-compliant. This filtering process results in 2237 qualified queries paired with 4474 trajectories, which serve as the training data for the Agentic VRPO phase.

- 5.1.2 Evaluation Metrics. During evaluation, we observe that the outputs of search agents are typically long. Specifically, even when the model answers the question correctly, it often includes extensive supplementary information. As noted in prior work [28, 32], this behavior makes exact-match metrics such as EM unsuitable for our

setting. Following [28, 32], we adopt accuracy (ACC𝑅) as our primary evaluation metric, which checks whether the golden answer is contained in the predicted answer generated by the search agent. To further refine our evaluation, we employ an LLM-as-Judge protocol [14] using Doubao-seed-1.8 as the judge model to determine whether the predicted answer is correct, denoted as ACC𝐿.

- 5.1.3 Baselines. To verify the effectiveness of DLLM-Searcher in enhancing the reasoning and information seeking capabilities of dLLMs, We compared DLLM-Searcher against several baselines: Traditional RAG : SuRe [13] executes multiple reasoning paths

- Table 1: Performance comparisons between Dllm-Searcher and the baselines on QA benchmarks. The best and second best results are bold and underlined, respectively; ‘†/‡’ represents in-domain/out-of-domain datasets; ‘/’ implies that the model struggles to generate valid tool call instructions, resulting in parsing failures; ‘∗’ means that the results were obtained under a modified experimental setup explained in § 5.1.3).

HotpotQA† 2Wiki† Bamboogle‡ Musique† Avg

Models

𝐴𝐶𝐶𝑅 𝐴𝐶𝐶𝐿 𝐴𝐶𝐶𝑅 𝐴𝐶𝐶𝐿 𝐴𝐶𝐶𝑅 𝐴𝐶𝐶𝐿 𝐴𝐶𝐶𝑅 𝐴𝐶𝐶𝐿 𝐴𝐶𝐶𝑅 𝐴𝐶𝐶𝐿

##### Traditional RAG

SuRe 32.4 48.4 22.2 26.8 17.6 28.0 7.2 10.0 19.9 28.3 Selective-Context 33.2 43.4 27.4 29.6 15.2 20.8 5.8 8.8 20.4 25.7 Adaptive-RAG 38.0 47.4 27.8 25.8 21.6 25.0 7.2 11.6 23.7 27.5 IRCoT 48.8 55.2 41.0 38.6 32.0 39.2 11.6 15.8 33.4 37.2 Iter-RetGen 41.6 54.4 32.4 34.4 26.4 32.0 14.8 18.2 28.8 34.8 CR-Planner 44.4 33.6 48.2 22.0 35.2 34.4 12.2 11.4 35.0 25.4 ReARTeR 46.8 50.6 55.4 53.4 49.6 54.4 29.6 30.2 45.4 47.2

##### ARM-based LLMs Agent

Search-o1 40.8 53.2 47.0 51.2 49.6 52.0 15.2 19.0 38.2 43.9 Search-R1 49.6 62.2 46.0 50.0 47.2 56.0 28.0 26.0 42.7 48.6 WebSailor∗ 50.4 52.4 59.4 61.4 57.6 65.6 22.0 28.0 47.4 51.9 R1Searcher∗ 58.0 62.2 59.6 63.4 66.4 68.8 28.2 31.4 53.1 56.5

##### dLLMs Agent

SDAR / / / / / / / / / / Dream 11.0 11.6 13.6 12.0 12.0 13.6 3.8 3.2 10.1 10.1 LLaDA 36.0 32.8 42.0 38.8 46.4 42.4 15.2 15.8 34.9 32.5 DLLM-Searcher 60.4 62.4 69.8 64.6 68.8 69.6 29.0 29.8 57.0 56.6

in parallel for a single query. Selective-Context [19] compresses retrieved documents to reduce context length. Adaptive-RAG [36] dynamically selects retrieval strategies depending on the complexity of the query. RAG-CoT methods, such as IRCoT [35], IterRetGen [26].CR-Planner [18], ReARTeR [32] scales RAG at inference time using Monte Carlo Tree Search (MCTS).

LLMAgents:Search-o1[17]integratesRAGwith Chain-of-Thought

(CoT) reasoning via prompt engineering. For models that leverage reinforcement learning (RL) to autonomously learn retrieval behaviors during inference, we include Search-r1 [11], WebSailor [15], and R1Searcher [28] as baselines. Note that WebSailor was trained with two tools, namely search and visit. To ensure consistency across all evaluations, we only equip it with the search tool in our experiments. R1Searcher was trained using a local search tool, the results reported correspond to the higher performance achieved between evaluations with the local search and Google Search.

dLLM Agents : To quantitatively benchmark the intrinsic performance of dLLMs in agentic tasks, we directly evaluate the dLLM backbone SDAR [6], Dream [41], LLaDA [22] using the standard ReAct paradigm, what’s more, for LLaDA and Dream, we use FastdLLM [38] to accelerate the inference.

5.1.4 Implementation Details. Model and Tools. We employ the SDAR model with a block size of 64 as our backbone. For the retrieval component, we utilize Google Search as our external tool, retrieving the top 10 search results.

Agentic SFT. During the SFT stage, we utilize an attention mask with a block size of 128. The training is conducted with a learning rate of 1𝑒−5, a total batch size of 32, and for 3 epochs.

Agentic VRPO. In the VRPO stage, we maintain an attention mask block size of 128. The model is trained with a learning rate of 5𝑒−7 and a batch size of 16 for 5 epochs.

Decoding Configuration. For both the VRPO rollout data generation and final evaluation, we apply our proposed P-ReAct strategy. We set the confidence bias to 0.5 and employ a low-confidence static approach for the remaining decoding steps. The inference configuration consists of 128 denoising steps, a block size of 128, and a temperature of 1.0.

Hardware.All experimentsandtrainingprocesses are implemented using the PyTorch framework and conducted on a server equipped with 8 × NVIDIA H100 GPUs.

### 5.2 Overall Performance

Table1 shows the results of DLLM-Searcher and the baselines on four mutil-hop QA benchmarks. We can obtain the following observations:

- Table 2: Performance comparison between Agentic SFT and Agentic VRPO on different datasets

Method

Dataset Metric

SFT VRPO HotpotQA

𝐴𝐶𝐶𝑅 57.2 60. 4 (+3.2) 𝐴𝐶𝐶𝐿 58.8 62. 4 (+3.6)

𝐴𝐶𝐶𝑅 66.4 69. 8 (+3.4) 𝐴𝐶𝐶𝐿 61.6 64. 6 (+3.0)

2Wiki

𝐴𝐶𝐶𝑅 64.6 68. 8 (+4.2) 𝐴𝐶𝐶𝐿 64.0 69. 6 (+5.6)

Bamboogle

𝐴𝐶𝐶𝑅 24.4 29. 0 (+4.6) 𝐴𝐶𝐶𝐿 26.6 29. 8 (+3.2)

Musique

1. DLLM-Searcher achieved performance improvements on multi-hop QA. Our method, DLLM-Searcher, achieves excellent performance across all multi-hop QA benchmarks under both the 𝑨𝑪𝑪𝑹 and 𝑨𝑪𝑪𝑳 metrics.

(1) It outperforms traditional RAG strategies by a substantial margin, especially attaining an improvement of about 19% over ReARTeR which is a strong baseline that leverages a PRM model to supervise the reasoning process.(2) It yields significant performance gains compared with vanilla dLLMs without any agentic post-training.(3) It achieves comparable performance against search agents built on ARMs, with the only performance gap observed on the Musique dataset relative to R1Searcher.

These results demonstrate that our two-stage post-training strategy tailored for dLLMs effectively enables the model to perform accurate and timely retrieval invocations throughout the reasoning process, thereby enhancing overall performance.

2. Maintaining Generalization Ability. Despite being trained on only 8k samples, DLLM-Searcher achieves strong performance on in-domain datasets such as HotpotQA, 2WikiMultiHopQA, and Musique, while also demonstrating impressive generalization capability on out-of-domain datasets such as bamboogle. This indicates that the model has effectively learned to integrate the retrieval of relevant documents with its internal reasoning process during training. Such an ability ensures the model’s robust performance on unseen datasets that require external information retrieval.

Furthermore, all results of DLLM-Searcher presented in Table 1 are obtained using the P-ReAct paradigm, which achieves substantial inference efficiency gains over the standard ReAct paradigm with negligible performance degradation. More results are reported in Section § 5.3.2.

### 5.3 Empirical Analysis

We conducted experiments to analyze the components of DLLMSearcher and answer the following research questions.

- 5.3.1 RQ1: Effectiveness of Post-training. We first evaluate the instruction-following capabilities of the vanilla SDAR model in agentic scenarios. We find that the model exhibits an almost complete inability to adhere to the rigid formatting protocols required for agentic interaction.

Table 3: Error Type Classification and Statistics

Error Type Count Percentage Empty Output 156 31.20% No tool_call 142 28.40% think Format Error 89 17.80% tool_call Format Error 35 7.00%

Specifically, when tested on 500 samples from the HotpotQA dataset, the vanilla SDAR model failed to complete a single successful interactions. All test cases were terminated during the first turn of the ReAct process due to formatt errors. We have summarized the 4 most frequently occurring types of errors: 1. Empty Output: The model directly outputs the end token "<|im_end|>" without generating any response content. 2. No tool_call: The model generates reasoning process within the <think> tag but fails to produce the required tool_call afterwards. 3. think Format Error: The model produces incomplete or malformed think tags, such as unclosed tags like <th. 4. tool_call Format Error: The model generates tool_call with incorrect JSON syntax or malformed function call structure like <tools>. The detailed statistics are presented in Table 3.

We attribute this catastrophic failure to the absence of instruction data tailored for tool-use and multi-step reasoning during the SDAR’s Continue Pre-Train (CPT) phase. Without targeted alignment specific to agentic workflows, the model suffers from severe structural hallucinations.

As reported in Table 2, Agentic SFT effectively rectifies these behavioral flaws, enabling SDAR to strictly follow the requisite tool_call formats. Consequently, the reasoning trajectories are no longer prematurely terminated early by parsing failures, allowing the model to successfully execute multi-step logic chains. The quantitative results across all four datasets confirm that the post-SFT model has acquired fundamental information seeking and reasoning capabilities. Furthermore, the subsequent Agentic VRPO stage delivers additional performance gains, consistently enhancing the model’s capabilities and yielding universal improvements across all benchmarks. Remarkably, both 𝐴𝐶𝐶𝑅 and 𝐴𝐶𝐶𝐿 exhibit gains exceeding 3 percentage points.

5.3.2 RQ2: Inference Efficiency. We evaluate our final model using both ReAct and P-ReAct. Under ReAct, we impose no additional constraints and allow the model to follow the standard Reasoning-Action-Observation cycle until reaching the maximum number of turns or producing a final answer. Under P-ReAct, we constrain each turn to complete the think and tool_call region within a single block, by pre-filling the <tool_call> boundary tokens and applying a confidence bias to guide decoding. As shown in the Figure 3, across the four datasets, P-ReAct achieves inference time reductions of 14.77%, 21.00%, 22.08%, and 12.67% relative to ReAct, with almost no performance degradation. These results indicate that P-ReAct effectively exploits the properties of dLLMs to prioritize decoding high-quality tool calls, and further accelerates search agent inference by overlapping reasoning with the waiting time for tool responses.

[Figure 40]

###### Figure 3: Accuracy and average end-to-end latency comparison of DLLM-Searcher under P-ReAct and ReAct paradigms on multi-hop QA tasks.

[Figure 41]

###### Figure 4: Comparison of accuracy changes on multi-hop QA tasks between DLLM-Searcher and Qwen3 series models when switching from ReAct to P-ReAct.

5.3.3 RQ3: Advantages of dLLMs’ Order-free Generation. Constrained by the causal attention mechanism and the next-token prediction paradigm, LLMs can only achieve parallel “thinking while waiting for tool response” capabilities comparable to P-ReAct in a training-free setting by restructuring each ReAct output into the sequence [<tool_call>...</tool_call> <think>...</think>]. We conducted experiments using three models of varying sizes from the Qwen3 series : Qwen3-8B, Qwen3-30B-A3B, and Qwen3235B-A22B, by modifying the prompts to instruct the models to generate the tool_call component first. As shown in Figure 4, ARMs can only achieve this behavior at the cost of notable performance degradation. In contrast, DLLM-Searcher with P-ReAct even achieves accuracy gains over standard ReAct on the HotpotQA and Musique datasets, with only minor accuracy losses observed on 2Wiki and Bamboogle. Overall, its performance degradation is far less significant than that of ARMs.

This experiment further demonstrates that ARMs rely heavily on explicitly decoded think segments to generate high-quality

tool_call instructions. In contrast, although DLLM-Searcher with P-ReAct ostensibly decodes the tool_call component first, the quality of the generated tool calls remains well-preserved. As discussed in § 4.4, diffusion models “know the answer before decoding it” they leverage bidirectional attention within blocks such that, even when the think region is not explicitly decoded, the model still utilizes the relevant reasoning information during the tool_call decoding process. In essence, each decoding step implements a form of latent reasoning, enabling robust tool call generation without explicit chain-of-thought verbalization.

### 5.4 Case Study

In this section, we present a real case to show how DLLM-Searcher enhances the performances. Specifically, we showed the first thinking and tool calling iteration of both the vanilla model and DLLMSearcher on a HotpotQA query: "In the 1973 NFL season, the Pro Bowl took place at what football stadium that is part of the Truman Sports Complex, along with what other stadium?". As observed, the vanilla model outputs an incorrect special token <tool>, whereas DLLM-Searcher produces the correct format. Furthermore, at timestep 32, only the tool_call region is fully decoded, while the think region remains entirely masked. The think region is not fully populated until the final stage of decoding. These observations demonstrate that our two-stage posttraining procedure, combined with the P-ReAct agent paradigm, significantly enhances the model’s instruction-following capability. Additionally, both reasoning and information-seeking abilities are notably improved, and the model successfully prioritizes the decoding of the tool_call component. This further validates the effectiveness of our proposed P-ReAct strategy.

###### Vanilla Model’s One ReAct Iteration

First, let’s identify the Pro Bowl’s location for 1973 and also verify the stadium, so we can compare it with the Truman Sports Complex. We’ll perform our search for these details using our search tool.\n\n <tool>\n {"name": "search", "arguments": {"query": ["Pro Bowl 1973 location", "Truman Sports Complex arenas"]}}\n </tool>\n <|im_end|>\n<|endoftext|>

###### DLLM-Searcher’s One P-ReAct Iteration

<think> \n To answer the question, I need to find the location of the 1973 NFL Pro Bowl and confirm if it’s within the Truman Sports Complex. Then, I need to identify the other stadium in that complex. First, I’ll search for the 1973 NFL’s Pro Bowl location and check if it’s in the Truman Sports Complex. If so, I’ll search for the other stadium in that complex to provide the full answer. \n </think>\n \n <tool_call> \n { "name": "search", "arguments": { "query": [ "1973 NFL Pro Bowl location", "Truman Sports Complex Pro Bowl stadiums list" ] } \n } </tool_call> <|im_end|>

###### Decoding Order

35 33 37 39 38 42 41 40 43 44 45 55 58 57 46 47 48 49 51 50 53 54 52 56 59 64 65 66 67 63 62 61 60 68 74 69 70 72 71 73 78 75 76 77 88 79 80 81 82 83 87 84 85 86 95 91 92 94 93 96 97 90 89 98 99 109 100 102 101 108 107 105 104 103 106 112 123 110 111 113 114 115 116 117 121 118 125 120 119 126 1 3 2 5 4 7 6 17 8 16 9 10 12 13 15 14 18 20 19 21 26 24 25 22 23 28 29 30 31 11 3 1 27 124

### 6 Conclusion

This paper presents DLLM-Searcher to bridge the gap between Diffusion Large Language Models and practical Search Agents. Two major obstacles that prevent this adoption are analyzed: the Agent Ability Challenge stemming from the dLLM backbone, and the Latency Challenge arising from the conventional ReAct paradigm. To address these challenges, we propose a tailored two-stage posttraining pipeline comprising Agentic SFT and Agentic VRPO, along with a novel P-ReAct paradigm. In this way, DLLM-Searcher enables dLLMs to keep thinking while waiting during external tool execution. Experimental results on four benchmarks demonstrate that DLLM-Searcher achieves approximately 15% inference acceleration over the conventional ReAct paradigm while maintaining comparable performance to mainstream ARM-based search agents, verifying the potential of dLLMs as efficient agent backbones and the effectiveness of parallelizing agentic reasoning and acting.

“We actually start to act before we are aware of our decision to do so.” — Ray Kurzweil, How to Create a Mind

### 7 Acknowledgements

We thank the group of Dr. Biqing Qi at Shanghai AI Lab for their work on SDAR: an excellent BDLM. We also appreciate the guidance from Dr. Biqing Qi and Shuang Chen during the SFT stage: modifying the block attention to use a block size of 128 and then training normally. We also thank Jinhao Jiang, Huatong Song and Peitian Zhang for their valuable insights on the training of search agents.

### References

- [1] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025.
- [2] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 17981–17993, 2021.
- [3] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. Llada2.0: Scaling up diffusion language models to 100b, 2025.
- [4] ByteDance. Doubao-seed-1.8, 2025.
- [5] Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, and A. Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, 2022.
- [6] Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, and Bowen Zhou. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation, 2025.
- [7] DeepMind. Gemini diffusion, 2025.

- [8] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024.
- [9] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.
- [10] Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov. Mercury: Ultra-fast language models based on diffusion, 2025.
- [11] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025.
- [12] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. Flashrag: A modular toolkit for efficient retrieval-augmented generation research. arXiv preprint arXiv:2405.13576, 2024.
- [13] Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, and Jinwoo Shin. Sure: Summarizing retrievals using answer candidates for open-domain QA of LLMs. In The Twelfth International Conference on Learning Representations, 2024.
- [14] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-as-judges: A comprehensive survey on llm-based evaluation methods. arXiv preprint arXiv:2412.05579, 2024.
- [15] Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025.
- [16] Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, and Shiwei Liu. Diffusion language models know the answer before decoding, 2025.
- [17] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models, 2025.
- [18] Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao, Shafiq Joty, and Lidong Bing. Can we further elicit reasoning in llms? critic-guided planning with retrieval-augmentation for solving challenging tasks. arXiv preprint arXiv:2410.01428, 2024.
- [19] Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. Compressing context to enhance inference efficiency of large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6342–6353, Singapore, December

2023. Association for Computational Linguistics.

- [20] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, ICML, 2024.
- [21] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data, 2023.
- [22] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.
- [23] Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, and Chongxuan Li. Principled rl for diffusion llms emerges from a sequence-level perspective, 2025.
- [24] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. ArXiv preprint, abs/2406.03736, 2024.
- [25] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711, 2023.
- [26] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. arXiv preprint arXiv:2305.15294, 2023.
- [27] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation, 2021.
- [28] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning, 2025.
- [29] Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher++: Incentivizing the dynamic knowledge acquisition of llms via reinforcement learning, 2025.
- [30] Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. Smart-searcher: Incentivizing the dynamic knowledge acquisition of LLMs via reinforcement learning. In Findings of the Association for Computational Linguistics: EMNLP

- 2025. Association for Computational Linguistics, November 2025.
- [31] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, YaQin Zhang, Yonghui Wu, and Hao Zhou. Seed diffusion: A large-scale diffusion language model with high-speed inference, 2025.
- [32] Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Yang Song, and Han Li. Rearter: Retrieval-augmented reasoning with trustworthy process rewarding. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’25, 2025.
- [33] MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Wenhan Dou, Yue Deng, Yunjie Fu, Junqi Ge, Chenxia Han, Tammy Huang, Zhenhang Huang, Jerry Jiao, Shilei Jiang, Tianyu Jiao, Xiaoqi Jian, Lei Lei, Ruilin Li, Ryan Luo, Tiantong Li, Xiang Lin, Ziyuan Liu, Zhiqi Li, Jie Ni, Qiang Ren, Pax Sun, Shiqian Su, Chenxin Tao, Bin Wang, Hellen Wang, Haonan Wang, James Wang, Jin Wang, Jojo Wang, Letian Wang, Shizun Wang, Weizhi Wang, Zixuan Wang, Jinfan Xu, Sen Xing, Chenyu Yang, Hai Ye, Jiaheng Yu, Yue Yu, Muyan Zhong, Tianchen Zhao, Xizhou Zhu, Yanpeng Zhou, Yifan Zhang, and Zhi Zhu. Mirothinker: Pushing the performance boundaries of open-source research agents via model, context, and interactive scaling, 2025.
- [34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
- [35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive

- multi-step questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10014–10037, 2023.
- [36] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. Self-knowledge guided retrieval augmentation for large language models. arXiv preprint arXiv:2310.05002, 2023.
- [37] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm, 2025.
- [38] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding, 2025.
- [39] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
- [40] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023.
- [41] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models, 2025.
- [42] Lingxiao Zhao, Xueying Ding, Lijun Yu, and Leman Akoglu. Improving and unifying discrete&continuous-time discrete denoising diffusion. ArXiv preprint, abs/2402.03701, 2024.
- [43] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference optimization for large language diffusion models, 2025.

