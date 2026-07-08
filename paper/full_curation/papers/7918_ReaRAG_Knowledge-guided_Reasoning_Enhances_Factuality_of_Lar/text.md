# arXiv:2503.21729v3[cs.CL]19May2025

## ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation

### Zhicheng Lee1, Shulin Cao1, Jinxin Liu1, Jiajie Zhang1, Weichuan Liu2, Xiaoyin Che2, Lei Hou1, Juanzi Li1 1Tsinghua University, 2Siemens AG

[Figure 1]

https://github.com/THU-KEG/ReaRAG

### Abstract

Large Reasoning Models (LRMs) exhibit remarkable reasoning abilities but rely primarily on parametric knowledge, limiting factual accuracy. While recent works adopt reinforcement learning (RL) training to integrate reasoning with retrieval, such methods are often complex and resource-heavy. Instead, our approach achieves comparable performance via strategic distillation, without the need for costly RL training. In this paper, we propose ReaRAG, a factuality-enhanced reasoning model that explores diverse queries without excessive iterations. Our solution includes a novel data construction framework with an upper bound on the reasoning chain length. Specifically, we first leverage an LRM to generate deliberate thinking, then select an action from a predefined action space (Search and Finish). For Search action, a query is executed against the RAG engine, where the result is returned as observation to guide reasoning steps later. This process iterates until a Finish action is chosen. Benefiting from ReaRAG’s strong reasoning capabilities, our approach outperforms existing baselines overall. Further analysis highlights its strong reflective ability to recognize errors and refine its reasoning trajectory. Our study enhances LRMs’ factuality while effectively integrating robust reasoning for RetrievalAugmented Generation (RAG).

### 1 Introduction

Large Reasoning Models (LRMs) such as OpenAI’s o1 (Jaech et al., 2024), Qwen’s QwQ32B1 (Team, 2024), GLM-Z12 and DeepSeek-R1 (DeepSeek-AI et al., 2025) demonstrate impressive reasoning capabilities on complex tasks (Xu et al., 2025). However, their reliance on parametric knowledge during reasoning limits performance

- 1For simplicity, all mentions of QwQ-32B in this paper refer to QwQ-32B-Preview.
- 2https://open.bigmodel.cn/dev/api/ Reasoning-models/glm-z1

[Figure 2]

[Figure 3]

|Question|
|---|

Iterative

|Final Answer|
|---|

|Reasoning Chain|
|---|

ReaRAG-9b

|RAG Engine<br><br>|Retrieved Docs| |
|---|---|
| | |
<br><br>|LLM|
|---|
<br><br>[Figure 4]|
|---|

|Internal Reasoning<br><br><think>...</think>|
|---|

|Final Answer|
|---|

|Question|
|---|

[Figure 5]

LRM

Figure 1: Unlike LRMs, ReaRAG iteratively constructs knowledge-guided reasoning chains for factual answers.

on question answering (QA) tasks that require factual answers, where reasoning beyond the model’s parametric knowledge is required.

To enhance LRMs’ factuality, RetrievalAugmented Generation (RAG) (Lewis et al., 2020; Shi et al., 2024; Guu et al., 2020) offers a promising solution by integrating external knowledge but faces challenges in retrieving relevant documents, which requires formulating precise search queries (Chan et al., 2024). Prior research has explored iterative retrieval strategies (Press et al., 2023; Shao et al., 2023), which construct reasoning chains of sub-queries and sub-answers to solve multi-hop QA. However, these methods suffer from error propagation, where mistakes in earlier steps mislead subsequent retrieval and reasoning, ultimately degrading the overall answer quality (Cao et al., 2023).

To address this limitation, Search-o1 (Li et al., 2025) prompts a LRM to iteratively retrieve documents and generate sub-answers via a Reason-inDocuments module, relying heavily on the model’s inherent reasoning and instruction-following capabilities. This leads to three key issues: (1) unreliable retrieval token generation, (2) information extraction failures and hallucinations during document reasoning, and (3) overthinking in multi-hop QA (Chen et al., 2024). Effectively integrating reasoning with retrieval remains challenging.

In response to this challenge, recent works such as R1-Searcher (Song et al., 2025) and Re-

Search (Chen et al., 2025) adopt reinforcement learning (RL) to incentivize Large Language Models (LLMs) to perform reasoning before retrieval. However, such policy optimization approaches involve complex training paradigms and require extensive computational resources.

In this paper, we propose ReaRAG, a factualityenhanced reasoning model for RAG that achieves comparable performance without relying on complex RL-based training. Instead of resource-heavy policy optimization, we strategically distill the reasoning capabilities of LRMs to iteratively construct knowledge-guided reasoning chains, which are then curated into a dedicated dataset with restricted chain lengths to fine-tune ReaRAG under the Thought-Action-Observation paradigm. During inference, ReaRAG iteratively performs the search action and dynamically decides when to trigger the finish action, avoiding excessive retrieval. Guided by external knowledge, ReaRAG reflects on its reasoning trajectory, detects errors, and realigns its reasoning toward the correct path, leading to improved performance with greater training efficiency and practical implementation.

To validate our approach, we conduct experiments on six benchmarks covering both multi-hop and single-hop QA, where ReaRAG achieves the highest overall performance. In summary, our contributions are as follows:

- • Enhancing LRMs’ factuality through knowledge-guided reasoning chain. We propose ReaRAG-9B, a model fine-tuned on a dedicated dataset to perform knowledge-guided reasoning, enabling reliable access to external knowledge. Analyses show that ReaRAG reflects on prior steps, strategically uses external knowledge to identify errors and refines its reasoning, showcasing robust multi-step reasoning capabilities.
- • Effectively combining strong reasoning with RAG. Compared to Search-o1, our model avoids redundant retrieval in multi-hop QA. Compared to RL-based methods, we offer a more efficient and robust solution by strategically distilling reasoning capabilities from a LRM, achieving comparable performance even without complex RL training.
- • Enhanced benchmark performance. Compared to RL-based methods, ReaRAG performs on par with or surpasses them on MuSiQue (Trivedi et al., 2022), HotpotQA (Yang et al., 2018), IIRC (Ferguson et al., 2020) and Natural Questions (NQ)

(Kwiatkowski et al., 2019), while outperforming them on harder tasks such as FRAMES (Krishna et al., 2024) and FanOutQA (Zhu et al., 2024).

### 2 Related Work

Reasoning-enhanced LLMs. Numerous works have investigated how to elicit the reasoning abilities of LLMs. Early approaches, such as Chainof-Thought (COT) (Wei et al., 2022), ReAct (Yao

- et al., 2023b) and Tree of Thought (ToT) (Yao et al.,

- 2023a) prompts LLMs to generate human-like step by step reasoning chain, though they still struggle on complex reasoning tasks. Recent advances of LRMs have scaled up CoT through RL (Kumar

et al., 2024), enabling models to generate long CoT before providing a final answer. Notably, LRMs include OpenAI’s o1 (Jaech et al., 2024), Qwen’s QwQ-32B (Team, 2024), GLM-Z1 and DeepSeekR1(DeepSeek-AI et al., 2025) demonstrate impressive performance in complex reasoning tasks. Despite these advancements, most LRMs lack the ability to interact with external knowledge, limiting their capacity to generate factual responses.

Retrieval-Augmented Generation. RAG has emerged as a promising paradigm for improving LLMs’ factuality. Early methods rely on a single retrieval step (Lewis et al., 2020; Borgeaud et al., 2022; Izacard et al., 2023), which often falls short for multi-hop QA tasks due to limited retrieval quality. To mitigate noisy retrieval (Shi et al., 2023), Self-RAG (Asai et al., 2024) and CRAG (Yan et al.,

- 2024) introduce reflection mechanisms on the retrieved documents, yet single-step retrieval remains insufficient. To address these limitations, iterative methods such as Iter-RetGen (Shao et al., 2023) and Self-Ask (Press et al., 2023) progressively retrieve relevant documents to gather sufficient information for final answer. Building on this, SearChain (Xu et al., 2024) first generates a complete reasoning chain, then verifies the answer of each node in the chain based on external knowledge. However, these approaches lack a strong reflection mechanism to recover from earlier reasoning errors.

Reasoning-enhanced RAG. Recent studies in scaling token generation at test-time (OpenAI, 2024; Muennighoff et al., 2025; Snell et al., 2024) to enhance LLMs’ reasoning capabilities have spurred interest in reasoning-enhanced RAG. RAGStar (Jiang et al., 2024) leverages Monte Carlo Tree Search to iteratively decompose multi-hop ques-

###### Data Construction

###### Fine-tuning

|RAG Engine<br><br>||Doc 1|
|---|
<br><br>|Doc 2|
|---|
<br><br>|Doc 3|
|---|
<br><br>Retrieved Docs| |
|---|---|
| |Document reasoning|
<br><br>|[Figure 6]<br><br>Answer generation|
|---|
|
|---|

Multi-hop Complex Questions

|Reasoning Chains| |
|---|---|
| | |

|Filtered Reasoning Chains|SFT|
|---|---|
| | |

[Figure 7]

LRM

[Figure 8]

[Figure 9]

Iterative

Subquery

|Reasoning Chains|
|---|

ReaRAG-9B

Subanswer

###### Inference

|Reasoning Chain<br><br>Thought 1: I need to find out ...<br><br>Action 1: Search (Who is the screenwriter of The Actress?)<br><br>Obs 1: Ruth Gordon<br><br>Thought 2: So, the screenwriter of "The Actress" is Ruth Gordon. Now, I need to ...<br><br>Action 2: Search (Who was Ruth Gordon married to?)<br><br>Obs 2: Ruth Gordon was married to Garson Kanin.<br><br>RAG Engine<br><br>RAG Engine<br><br>Thought 3: ... I should verify if there was only one spouse or...<br><br>Action 3: Search (Was Ruth Gordon married more than once?)<br><br><br><br><br>Obs 3: Based on the given Contexts, the information about Ruth Gordon's marriage is not provided...<br><br>Thought 4: ... according to the initial search, she was married to Garson Kanin.... Action 4: Finish (Garson Kanin is the spouse of the screenwriter...)<br><br>RAG Engine<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

[Figure 13]

###### Who is the spouse of the screenwriter of The Actress?

[Figure 14]

ReaRAG-9b

Iterative

|Answer Generation<br><br>|Search query: Who is the screenwriter of The Actress Search result: Ruth Gordon<br><br>... Search query: Was Ruth Gordon married more...? Search result: Based on the given Contexts, ...<br><br>|[Figure 15]<br><br>Answer generation|
|---|
<br><br>Generate short answer based on gathered information|
|---|
<br><br>|
|---|

Final Answer

###### Garson Kanin

- Figure 2: Overview of our approach to develop a factuality-enhanced reasoning model ReaRAG. To equip ReaRAG with knowledge-guided reasoning ability, we propose an automated data construction approach (Algorithm 1). Next, we fine-tune ReaRAG on the constructed dataset to conduct reasoning iteratively, following the Thought-ActionObservation Paradigm to solve complex queries. Pseudocode for the inference stage is provided in Algorithm 2.

tions, guided by a reward model during the tree expansion. Search-o1 (Li et al., 2025) introduces a document reasoning module and prompts QwQ32B to generate special tokens that trigger knowledge retrieval. However, it heavily depends on the base model’s instruction-following and inherited reasoning capabilities, leading to three key challenges: unreliable generation of specialized tokens, failure in extracting information from retrieved documents, and overthinking for multi-hop QA. To integrate reasoning with retrieval, R1-Searcher (Song et al., 2025) and ReSearch (Chen et al., 2025) adopt a RL-based method that incentivizes LLMs to reason before retrieving. However, despite the complexity of RL-based training paradigms, our approach achieves comparable performance through strategic distillation, offering greater training efficiency and practical ease of implementation.

### 3 Methodology

In this section, we first formalize the task, then present our novel approach for developing ReaRAG as illustrated in Figure 2.

#### 3.1 Task formulation

We focus on the multi-hop QA task, where iterative reasoning improves answer accuracy. Given a

question x, our goal is to construct a knowledgeguided reasoning chain C to enhance the factual correctness of the generated answer yˆ. Specifically, the reasoning chain is formulated as a sequence of N steps, where each step consists of a reasoning thought τt, an action αt, and an observation ot:

C = {(τt,αt,ot)}Nt=1, 1 ≤ t ≤ Tmax (1)

The number of reasoning steps N is dynamically determined by the model but is constrained by an upper limit Tmax to prevent indefinite iterations, i.e., N ≤ Tmax. To guide the reasoning process with external knowledge, we define the action space as A = {search(),finish()} where search takes a search query as input, while the input to finish represents the derived answer. At each step t, the model refines a search query based on the reasoning thought τt and executes the search action to retrieve relevant information from the RAG engine R. The process continues until the model selects the finish action, at which point the final answer yˆ is derived from all prior reasoning steps. This ensures that the answer is grounded in retrieved knowledge through an iterative and structured reasoning process, thereby enhancing the factual reliability of LRMs.

- Algorithm 1 Data Construction Input: Seed Dataset Dseed, Large Reasoning Model MLRM, Instruction prompt Pd, Max iterations Tmax, RAG engine R Output: Dataset Dreason with reasoning chains

- 1: Initialize Dreason ← ∅
- 2: for each (xi,doci) ∼ Dseed do

▷ Sample question and gold documents

- 3: t ← 0 ▷ Iteration counter
- 4: Ci ← [] ▷ Reasoning chain
- 5: while t < Tmax do
- 6: yt′ ← MLRM([Pd ⊕ xi ⊕ Ci])

▷ Generate response

- 7: (τt,αt) ← parse(yt′)

▷ Extract thought τt and action αt

- 8: Get action type αttype ← αt[‘type’]
- 9: if finish ∈ αttype then
- 10: Append (τt,αt) to Ci
- 11: break
- 12: else if search ∈ αttype then
- 13: qs ← αt[‘query’]
- 14: ot ← R(qs,doci)

▷ Get ot from RAG engine

- 15: Append (τt,αt,ot) to Ci
- 16: end if
- 17: t ← t + 1
- 18: end while
- 19: Append Ci to Dreason
- 20: end for
- 21: return Dreason

- 3.2 Knowledge-guided reasoning chain generation

While existing LRMs demonstrate strong reasoning capabilities, they often fail to ground their reasoning process in factual knowledge. To make external knowledge accessible, we design a structured reasoning step where each step consists of a reasoning thought τt, an action αt, and an observation ot.

- • Reasoning thought τt: Represents the model’s thought process where it reflects on prior actions and observations before deciding an action and its input parameter.
- • Action αt: A JSON dictionary containing an action sampled from the action space A along with the corresponding input parameter.
- • Observation ot: Feedback received after executing action αt, guiding subsequent reasoning.

To equip ReaRAG with the ability to construct

reasoning chain guided by external knowledge, we propose an automated data construction approach, as detailed in Algorithm 1.

Data construction. Given a multi-hop question xi sampled from the seed dataset, we prompt the LRM MLRM with instruction prompt Pd (see Appendix B) to collect the reasoning thoughts and actions. Next, the search query is extracted and executed against RAG Engine R to obtain an observation ot. The process iterates until either model decides an finish action or the iteration count exceeds the maximum iterations Tmax.

Data filtering. Previous work studies have shown that the performance of the LLMs heavily depends on the quality of fine-tuning data (Gunasekar et al., 2023). To ensure high-quality reasoning chains, we apply data filtering by comparing the final answer yˆi derived from the reasoning chain Ci against the ground truth answer yi using the F1 metric. Reasoning chains with an F1 score of 0 are discarded to maintain data integrity.

#### 3.3 Factuality-enhanced LRMs: ReaRAG

Fine-tuning. To incorporate knowledge-guided reasoning ability into the model, we perform supervised fine-tuning (SFT) on the constructed dataset discussed in the previous section. Each sample is a sequence of conversation chain S = {P,xi,{(τt,αt,ot)}Nt=1}, where P denotes the instruction prompt (see Appendix B). We fine-tune our factuality-enhanced LRM, ReaRAG (MReaRAG) using loss function below:

##### L = −

j

1 × log MReaRAG(sj | s<j) (2)

where sj represents the textual tokens of the input sequence S, 1(·) is a loss mask indicator, which is set to True on thought τt and action αt tokens, ensuring that the loss is only computed over the tokens contributed to the thought reasoning as well as action, rather than the entire sequence.

Inference. After fine-tuning, ReaRAG is equipped with advanced reasoning capabilities to solve multi-hop QA. Given an instruction prompt P and a question x, the model first generates a reasoning thought τ0 and an initial action α0, typically a search action. The search query qs is extracted and executed over the RAG engine R, which returns an observation o0. The triplet

- Algorithm 2 Inference Input: Input question x, documents doc, ReaRAG MReaRAG, Answer LLM MAns, Instruction prompt P, Answer Prompt Pans, Max iterations Tmax, RAG engine R Output: Final answer, yˆ

- 1: t ← 0, C ← [] Sum ← [] ▷ Initialization
- 2: while t < Tmax do
- 3: yt′ ← MReaRAG([P ⊕ x ⊕ C])

▷ Generate response for iteration t

- 4: (τt,αt) ← parse(yt′)

▷ Extract thought τt and action αt

- 5: Get action type αttype ← αt[‘type’]
- 6: if finish ∈ αttype then
- 7: yˆ ← MAns([Pans ⊕ x ⊕ Sum])
- 8: return final answer yˆ
- 9: else if search ∈ αttype then
- 10: qs ← αt[‘query’]
- 11: ot ← R(qs,doc)

▷ Get ot from RAG engine

- 12: Append (τt,αt,ot) to C
- 13: Append (qs,ot) to Sum
- 14: end if
- 15: t ← t + 1
- 16: end while

(τt,αt,ot) is appended to the reasoning chain C, while the pair (qs,ot) is appended to the summary chain Sum. This process iterates until ReaRAG decides on a finish action at step N. Upon this action, we concatenate all gathered information in Sum and prompt an answer model MAns to generate final answer yˆ using the prompt Pans described in Appendix B. The pseudocode for the inference stage is provided in Algorithm 2.

### 4 Experiments

#### 4.1 Experimental setup

Dataset and metrics. We evaluate our approach on multi-hop QA benchmarks requiring reasoning across multiple documents, including MuSiQue (MQ) (Trivedi et al., 2022), HotpotQA (HP) (Yang et al., 2018) and IIRC (Ferguson et al., 2020), as well as recently proposed benchmarks, including FRAMES (FR) (Krishna et al., 2024) and FanOutQA (FQ) (Zhu et al., 2024). To evaluate single-hop reasoning capabilities, we additionally include NQ (Kwiatkowski et al., 2019). Since these datasets require open-ended answers, traditional metric such as exact match (EM) may fail to cap-

ture semantically equivalent responses (Yin et al., 2024). Hence, we include the LLM-as-a-Judge metric (ACCL) (Zheng et al., 2023) based on GPT4o for more accurate evaluation using the prompt Pjudge described in Appendix B. For FanOutQA, we follow the metric used in the original benchmark and report the Rouge-L (R-L) score instead of EM. We randomly sample 100 validation examples each from MuSiQue, HotpotQA, and NQ, 200 from IIRC, and 396 from FRAMES. For FanOutQA, we evaluate on the full development set, which contains 310 examples.

Baselines. We compare our approach against multiple baselines, categorized based on their access to external knowledge. These include incontext retrieval, where the corpus is directly appended to the language model’s context; vanilla RAG, which performs a single retrieval based on the original input question; and state-of-the-art advanced RAG methods proposed recently.

In the in-context settings, we use GLM-4-9B and GLM-4-32B (Zeng et al., 2024), both with a 128k context window. The vanilla RAG baselines employ the same models, along with QwQ32B (Team, 2024). For advanced RAG baselines, we include Self-RAG (Asai et al., 2024), which fine-tunes Llama-2-7B to retrieve documents on demand and filter noisy evidence; SearChain (Xu

- et al., 2024), which generates a Chain-of-Query structure enabling multi-turn retrieval with verification; and Search-o1 (Li et al., 2025), which proposes a framework for LRMs to perform iterative knowledge retrieval. We further consider R1Searcher (Song et al., 2025) and ReSearch (Chen
- et al., 2025), which incentivize LLMs to perform reasoning prior to retrieval through RL.

#### 4.2 Implementations details

Data construction and fine-tuning. The seed dataset described in Algorithm 1 is derived from the training sets of MuSiQue, HotpotQA, and NQ, with QwQ-32B as the LRM. To ensure model’s general capabilities, we fine-tune GLM-4-9B (Zeng et al., 2024) with the constructed dataset (roughly 20k filtered samples), as well as the general SFT dataset from GLM-4 (Zeng et al., 2024).

Evaluation. For MuSiQue and HotpotQA, we use the original corpora provided by the respective authors. For NQ, we follow the corpus setup from “Lost in the middle” (Liu et al., 2024). To increase the difficulty, particularly for comparisons with

Multi-hop Single-hop

Model

MuSiQue HotpotQA IIRC FRAMES FanOutQA NQ Average ACCL EM ACCL EM ACCL EM ACCL EM ACCL R-L ACCL EM ACCL

In-context

GLM-4-9B (128k) 23.5 15.0 58.0 47.0 - - - - - - 45.5 26.0 GLM-4-32B (128k) 33.5 17.0 65.5 50.0 - - - - - - 52.5 24.0 -

###### Vanilla RAG

GLM-4-9B (128k) 25.5 14.0 68.0 52.0 27.3 22.0 11.0 7.3 4.2 15.6 49.0 32.0 30.8 GLM-4-32B (128k) 29.0 17.0 67.5 52.0 26.0 14.5 13.9 7.6 3.9 12.3 53.0 39.0 32.2 QwQ-32B 36.0 20.0 67.0 47.0 32.8 25.0 23.4 13.9 9.7 14.2 48.0 26.0 36.1

###### Advanced RAG

Self-RAG-7B 24.0 13.0 45.5 31.0 21.8 12.0 8.8 5.6 3.9 14.2 40.0 28.0 24.0 SearChain-7B 8.5 3.0 23.0 17.0 25.8 18.5 16.0 10.6 4.8 17.7 28.0 13.0 17.7 Search-o1-32B 40.5 32.0 55.5 38.0 32.8 29.5 27.8 19.2 12.3 14.9 43.0 28.0 35.3 R1-Searcher-7B-Base 64.0 52.5 79.5 68.0 42.5 35.5 26.9 18.7 5.8 15.6 48.5 35.0 44.5 ReSearch-Qwen-7B-Instruct 65.0 49.0 72.5 59.0 41.8 33.5 29.4 21.0 6.8 15.8 53.0 37.0 44.7

Ours ReaRAG-9B 67.0 46.0 81.5 58.0 41.0 27.5 34.2 21.2 11.6 24.3 55.5 32.0 48.5

Table 1: Main experimental results across six benchmarks. Bold and underline indicate the best and second best results. We report the traditional EM scores as well as ACCL, a metric based on the LLM-as-a-Judge framework using GPT-4o. For FanOutQA, we follow the original benchmark and report the Rouge-L (R-L) score. Our model, ReaRAG-9B, achieves the best overall performance, demonstrating the effectiveness of our strategic distillation approach, even when compared to recent RL-based baselines that integrate reasoning with retrieval.

long-context LLMs, we scale up the number of distractor documents, resulting in corpora with 48k to 58k token lengths. Thus, this design demands high-quality queries to enhance retrieval quality.

For IIRC, FRAMES, and FanOutQA, where official corpora are incomplete or unavailable, we retrieve the top 5 most relevant Wikipedia snippets via the Google Serper API. These benchmarks are excluded from the in-context settings evaluations due to the lack of official corpora. All baselines are evaluated using their official implementations with our RAG engine and corpus setup.

RAG engine. Our RAG engine consists of two main components: retrieval and generation. For retrieval, we utilize the embedding model embedding-

- 3 from Zhipu’s API3, along with a reranker based on the GLM3 architecture to enhance retrieval quality. This setup is applied only to benchmarks with self-constructed corpora, while for IIRC, FRAMES, and FanOutQA, documents are retrieved directly via Google. For generation, we use GLM-4-32B with a 128k context window to generate responses based on the retrieved documents.

#### 4.3 Main results

Table 1 presents our main results across six benchmarks. Overall, our approach ReaRAG9B achieves the highest average ACCL (48.5)

3https://bigmodel.cn/dev/api/vector/embedding

across all evaluated methods. Despite the recent momentum around RL in RAG, particularly with models such as R1-Searcher and ReSearch, ReaRAG-9B trained solely with SFT to generate knowledge-guided reasoning chain, still matches or exceeds performance on MuSiQue, HotpotQA, IIRC, and NQ, and outperforms RL-based baselines on FRAMES and FanOutQA (by 5.8%-7.3% ACCL), suggesting SFT can achieve equally powerful performance for open-ended QA tasks. Although designed for multi-hop tasks, ReaRAG also performs strongly on single-hop QA, achieving 55.5 ACCL on NQ, demonstrating ReaRAG’s capabilities beyond multi-hop reasoning.

However, discrepancies between EM and ACCL, e.g., on NQ, where ReaRAG achieves a higher ACCL than ReSearch (55.5 vs. 53.0) but yields a lower EM score (32.0 vs. 37.0), suggest that EM may fail to capture contextually valid answers, a phenomenon consistently observed across all other benchmarks, necessitating the use of ACCL metric.

Comparing vanilla RAG with in-context settings across different GLM-4 backbone scales, we find that vanilla RAG generally performs better, suggesting that long-context models may struggle with distractor-heavy corpora. The only exception is MuSiQue with the GLM-4-32B backbone, where the in-context setting slightly outperforms vanilla RAG (33.50 vs. 29.00). Under vanilla RAG setting, QwQ-32B significantly outperforms GLM-4-

32B on complex multi-hop benchmarks such as FRAMES and FanOutQA, highlighting the advantage of LRMs with strong reasoning capabilities.

Notably, the original SearChain paper reports results using GPT-3.5, a proprietary large-scale model. For a fair comparison, we replace it with Qwen2.5-7B-Instruct (Yang et al., 2024). Under this setting, SearChain underperforms across all benchmarks. Similarly, Self-RAG, despite aiming to improve retrieval quality, lacks multi-turn retrieval strategies and shows weak performance. Search-o1, although leveraging QwQ-32B, a strong reasoning LRM to perform multi-turn retrieval, only performs on par with vanilla RAG settings. We further analyze these findings in Section 4.5.

#### 4.4 Ablation

Closed-book performance. We conduct a closed-book experiment to evaluate the parametric knowledge of the language models. The results, presented in Table 3 show that QwQ-32B outperforms GLM-4 on multi-hop benchmarks requiring strong reasoning, except for HotpotQA and FanOutQA. Nevertheless, their parametric knowledge remains insufficient compared to the results in Table 1.

Advantage of strong reasoning. To evaluate the impact of strong reasoning capabilities, we finetune a model that lacks such abilities under the same Thought-Action-Observation paradigm. This variant, denoted as w/o reasoning in Table 4, shares the same backbone architecture as ReaRAG-9B and follows the data construction process outlined in Algorithm 3. However, instead of leveraging a strong reasoning model like QwQ-32B for data generation, we employ GLM-4-9B, which lacks robust reasoning abilities. Unlike the previous data construction approach in Algorithm 1, which used only multi-hop questions as input, we now provide GLM-4-9B with additional information, including ground-truth decompositions and ground-truth answers. The instruction prompt Pablat used to generate its reasoning chain is detailed in Appendix B.

Table 4 shows that ReaRAG-9B with enhanced reasoning capabilities (w/ reasoning) consistently outperforms its counterpart without reasoning, achieving a notable gain of 6-12% ACCL gain on the multi-hop benchmarks and 3.5% gain on singlehop NQ. The smaller gain on NQ suggests that strong reasoning offers limited benefit for singlehop tasks, consistent with the assumption that such

questions require less compositional reasoning. 4.5 Analysis

Multi-hop Single-hop MQ HP IIRC FR FQ NQ

Invalid (%)19.028.0 18.0 18.930.0 25.0

Table 2: Invalid generation rates of special tokens in QwQ-32B, leading to retrieval failures in Search-o1.

#### 4.5.1 Performance against strong baseline

We conduct an in-depth analysis and compare our approach against the strong baselines, including Search-o1, R1-Searcher and ReSearch. Below, we highlight key factors affecting their performance.

Token generation failure. QwQ-32B struggles to follow the instruction prompt, failing to generate special tokens (e.g., |begin_search_query|) essential for Search-o1 to retrieve external knowledge. This limitation forces Search-o1 into a closed-book setting, significantly impairing its performance. Table 2 quantifies this issue, revealing invalid generation rates of 18–30%.

Information extraction failure. Search-o1 introduces the Reason-in-Documents module, leveraging QwQ-32B for in-depth reasoning over retrieved documents to generate responses as search results. However, this module has a key limitation: it may incorrectly conclude that no useful information is available (Table 5). Our analysis identifies the primary cause: the module attempts to answer the original multi-hop question based on the search query, but the retrieved information is insufficient. For example, as shown in Table 5, the module searches for information related to “Hannibal and Scipio book” at first search, but the retrieved content only includes the book’s author, lacking information about the place of education. This flaw weakens Search-o1, as it continuously searches for non-existent information, causing reasoning paths to diverge and ultimately hitting iteration limits.

Hallucination in Reason-in-Documents module. The Reason-in-Documents module is prone to hallucination Table 6. For instance, when searching for the members of Bruce Lee Band, the module fails to find relevant information and fabricates “Less Than Records” based on parametric knowledge rather than the provided corpus. This hallu-

| |Search-o1<br><br>Ours<br><br>R1-Searcher<br><br>ReSearch| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- 0
- 1
- 2
- 3
- 4
- 5
- 6

AverageChainLength

MuSiQue HotpotQA IIRC FRAMES FanOutQA NQ

- Figure 3: Comparison of chain length across all benchmarks. We measure the number of retrievals needed for

the baselines to achieve a full ACCL score. Search-o1 consistently requires more steps than ReaRAG, highlighting the tendency of the underlying LRM to overthink in multi-hop QA tasks.

cination propagates through subsequent reasoning steps, degrading the final answer quality.

Poor generalization on harder benchmarks. R1-Searcher and ReSearch show weaker performance on FRAMES and FanOutQA, despite matching ReaRAG on other benchmarks. Case studies in Tables 7 and 8 reveal that both models struggle to reason effectively in the presence of noisy information. R1-Searcher, in particular, proceeds to generate final answers despite lacking key information. These findings suggest that RL-based models lack the ability to reflect and revise earlier reasoning, an ability essential for robust multi-hop reasoning. In contrast, ReaRAG successfully realigns its reasoning path, even after early-stage errors, demonstrating greater robustness on complex multi-hop tasks.

Overthinking in multi-hop QA. Recent studies have shown that LRMs tend to overthink by generating overly long reasoning chains, causing redundancy in multi-hop QA (Chen et al., 2024; Team et al., 2025). We analyze this by comparing the number of retrievals needed to achieve full ACCL. Compared to Search-o1, ReaRAG requires fewer steps across all benchmarks except for FanOutQA, demonstrating efficiency in multi-hop reasoning while achieving better performance. Case studies in Tables 9 and 10 further highlight this difference. However, for baselines such as ReSearch and R1-Searcher which are optimized for retrieval, ReaRAG consistently takes more steps.

#### 4.5.2 Strength of ReaRAG

This section showcases ReaRAG’s advanced reasoning capabilities. Table 11 shows that ReaRAG initially mistakenly identified “Anne of Austria” as the grandmother of “Philippe” rather than his mother. However, ReaRAG later detected this mistake, verified the information, and corrected it. This self-correction mechanism helps prevent errors from propagating to later reasoning steps.

- Table 12 shows how ReaRAG resolves ambiguity in a multi-hop question via multiple retrieval. The query involves “The Hard Easy”, which refers to both a film and a TV series. At the sixth reasoning step, ReaRAG also faces conflicting information but successfully disambiguates and provides the correct answer.
- Table 13 provides another example of ReaRAG handling ambiguity in a multi-hop question while resolving a knowledge conflict. Its parametric knowledge incorrectly states that Sonic is voiced by “Roger Craig Smith” instead of “Jim Cummings”. ReaRAG detects and corrects this inconsistency, ultimately reaching the correct answer. This case further highlights its robust reasoning abilities.

These examples highlight ReaRAG’s ability to iteratively perform knowledge-guided reasoning. Compared to existing baselines, our approach better integrates reasoning model with external knowledge, enhancing factual accuracy.

### 5 Conclusion

In this study, we introduce ReaRAG, a factualityenhanced reasoning model capable of performing knowledge-guided reasoning. ReaRAG iteratively plans and reflects on reasoning steps, leveraging external knowledge to validate each step in the reasoning chain. Through comprehensive evaluation across six benchmarks, ReaRAG achieves the best overall performance, outperforming RLbased methods on challenging benchmarks such as FRAMES and FanOutQA, while matching them on easier tasks. Unlike RL-based approaches requiring complex policy optimization and high computational cost, our method strategically distills reasoning capabilities from LRMs, offering a more efficient alternative. Further analysis shows its robustness in handling complex multi-hop questions while mitigating the overthinking behavior seen in Search-o1. These results underscore the effectiveness of strategic reasoning distillation as a practical alternative to RL-based RAG methods.

### Limitations

Limited action space While ReaRAG demonstrates strong performance in the QA task, its action space is currently limited to only search and finish in this study. Consequently, it is restricted to processing local knowledge sources and cannot perform actions such as leveraging a code compiler for coding tasks, executing mathematical calculations, or conducting real-time web searches. Expanding its action space could enhance its adaptability across diverse problem domains.

Data construction efficiency To equip ReaRAG with a structured reasoning process, we fine-tune ReaRAG using structured responses generated by the LRM. However, this approach relies on the LRM’s strong instruction-following ability, and a substantial portion of the data is discarded due to validity issues, leading to computational inefficiency and resource waste. Improving data augmentation techniques could mitigate this limitation.

Inference latency ReaRAG solves questions iteratively, requiring multiple reasoning steps to reach the final answer. While this enhances accuracy, it also increases inference time compared to models that generate answers in a single pass. This trade-off between reasoning depth and efficiency may limit its practicality in real-time applications or scenarios with strict latency constraints.

### Acknowledgments

This work is supported by National Natural Science Foundation of China (62476150), Beijing Natural Science Foundation (L243006), Tsinghua University Initiative Scientific Research Program and Tsinghua University (Department of Computer Science and Technology)-Siemens Ltd., China Joint Research Center for Industrial Intelligence and Internet of Things (JCIIOT).

This paper involved the use of AI-assisted tools (e.g., ChatGPT) for language refinement and editing. All content was reviewed and verified by the authors.

### References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 2206–2240. PMLR.

Shulin Cao, Jiajie Zhang, Jiaxin Shi, Xin Lv, Zijun Yao, Qi Tian, Lei Hou, and Juanzi Li. 2023. Probabilistic tree-of-thought reasoning for answering knowledgeintensive complex questions. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 12541–

12560. Association for Computational Linguistics.

Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. 2024. RQ-RAG: learning to refine queries for retrieval augmented generation. CoRR, abs/2404.00610.

Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. 2025. Research: Learning to reason with search for llms via reinforcement learning. CoRR, abs/2503.19470.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2024. Do NOT think that much for 2+3=? on the overthinking of o1-like llms. CoRR, abs/2412.21187.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang,

R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948.

James Ferguson, Matt Gardner, Hannaneh Hajishirzi, Tushar Khot, and Pradeep Dasigi. 2020. IIRC: A dataset of incomplete information reading comprehension questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 1137–1147. Association for Computational Linguistics.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. 2023. Textbooks are all you need. CoRR, abs/2306.11644.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval augmented language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3929–3938. PMLR.

Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models. J. Mach. Learn. Res., 24:251:1–251:43.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit,

Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. 2024. Openai o1 system card. CoRR, abs/2412.16720.

Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie Wang, Wayne Xin Zhao, Yang Song, and Tao Zhang. 2024. Rag-star: Enhancing deliberative reasoning with retrieval augmented verification and refinement. CoRR, abs/2412.12881.

Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. 2024. Fact, fetch, and reason: A unified evaluation of retrievalaugmented generation. CoRR, abs/2409.12941.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. 2024. Training language models to selfcorrect via reinforcement learning. arXiv preprint arXiv:2409.12917.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics, 7:452– 466.

Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic searchenhanced large reasoning models. arXiv preprint arXiv:2501.05366.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393.

OpenAI. 2024. Learning to reason with llms.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 5687–5711. Association for Computational Linguistics.

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 9248–9274. Association for Computational Linguistics.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 31210–31227. PMLR.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: retrievalaugmented black-box language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 8371–8384. Association for Computational Linguistics.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. CoRR, abs/2408.03314.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and JiRong Wen. 2025. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. CoRR, abs/2503.05592.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang,

Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. 2025. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599.

Qwen Team. 2024. Qwq: Reflect deeply on the boundaries of the unknown.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Musique: Multihop questions via single-hop question composition. Trans. Assoc. Comput. Linguistics, 10:539–554.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, Chenyang Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Sijian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao, and Yong Li. 2025. Towards large reasoning models: A survey of reinforced reasoning with large language models. CoRR, abs/2501.09686.

Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. 2024. Search-in-the-chain: Interactively enhancing large language models with search for knowledge-intensive tasks. In Proceedings of the ACM on Web Conference 2024, WWW 2024, Singapore, May 13-17, 2024, pages 1362–1373. ACM.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.

2024. Corrective retrieval augmented generation. CoRR, abs/2401.15884.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 technical report. CoRR, abs/2412.15115.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and

Christopher D. Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2369–2380. Association for Computational Linguistics.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023b. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Raghavi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. 2024. Agent lumos: Unified and modular training for open-source language agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12380–12403. Association for Computational Linguistics.

Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. 2024. Chatglm: A family of large language models from GLM-130B to GLM-4 all tools. CoRR, abs/2406.12793.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. 2024. Fanoutqa: A multi-hop, multidocument question answering benchmark for large

language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, ACL 2024 - Short Papers, Bangkok, Thailand, August 11-16, 2024, pages 18–37. Association for Computational Linguistics.

### Appendix

### A Ablation

Multi-hop Single-hop

Model

MuSiQue HotpotQA IIRC FRAMES FanOutQA NQ Average ACCL EM ACCL EM ACCL EM ACCL EM ACCL R-L ACCL EM ACCL

GLM-4-9B(128k) 3.5 0.0 29.5 23.0 17.0 15.0 5.2 3.5 7.1 18.0 27.50 16.00 15.0 GLM-4-32B(128k) 6.5 1.0 40.0 28.0 17.0 11.5 10.5 4.6 11.9 22.5 44.5 25.0 21.7 QwQ-32B 11.0 2.0 35.0 10.0 21.3 14.5 14.4 9.1 11.0 13.2 37.5 12.0 21.7

- Table 3: Closed-book performance of language models on multi-hop and single-hop benchmarks. These models perform better on single-hop benchmarks but score significantly lower on multi-hop benchmarks, highlighting the limitations of relying solely on parametric knowledge for these benchmarks.

Model

Multi-hop Single-hop

MuSiQue HotpotQA IIRC FRAMES FanOutQA NQ Average ACCL EM ACCL EM ACCL EM ACCL EM ACCL R-L ACCL EM ACCL

ReaRAG-9B

- - w/o reasoning 57.5 38.0 69.5 50.0 31.8 22.0 26.6 19.2 5.5 21.6 52.0 29.0 40.5

- - w/ reasoning 67.0 46.0 81.5 58.0 41.0 27.5 34.2 21.2 11.6 24.3 55.5 32.0 48.5

- Table 4: Performance comparison of models with and without strong reasoning capabilities. w/ reasoning consistently outperforms w/o reasoning across all benchmarks, demonstrating the effectiveness of our fine-tuning process, which enables ReaRAG-9B to inherit the strong reasoning abilities of LRM.

Algorithm 3 Data Construction to fine-tune ReaRAG w/o reasoning Input: Seed Dataset Dseed, Large Language Model MLLM, Instruction Prompt Pablat Output: Dataset Dablat

- 1: Initialize Dablat ← ∅
- 2: for each (xi,yi,decompi) ∼ Dseed do

▷ Sample question xi, ground truth answer yi and golden decomposition decompi

- 3: ri′ ← MLLM([Pablat ⊕ xi ⊕ yi ⊕ decompi])

▷ Generate response

- 4: Ci = [{τt,αt,ot}Nt=1] ← parse(ri′)

▷ Parse a list of thought τt, action αt and observation ot into reasoning chain Ci

- 5: Append Ci to Dablat
- 6: end for
- 7: return Dablat

### B Prompts

Instruction prompts Pd for data construction to fine-tune ReaRAG

Your task is to solve a question answering task. To improve your solving accuracy, please conduct reasoning processes following this sequence: Thought, Action, Observation steps. Thought can reason about the current situation, and Action is in the form of a function. There are two available function types:

###### Available Functions:

- (1) Search

{

"name": "search", "description": "It can help you find useful information through the internet or local

knowledge base. You can use this tool to access external knowledge.",

"parameters": { "type": "object", "properties": {

"query": {

"description": "what you want to search" }

}, "required": ["query"]

} }

- (2) Finish

{

"name": "finish", "description": "You can use this function to make a conclusion from the reasoning process

and give the final answer. The reasoning process is completed after this ‘finish‘ function is called",

"parameters": { "type": "object", "properties": {

"answer": {

"description": "the final answer" }

}, "required": ["answer"]

}

} Some important rules you must follow:

- (1) Please follow the function calling format above strictly.
- (2) A set of ‘Thought‘, ‘Action‘, and ‘Observation‘ is considered as one reasoning step. Add numbering after each ‘Thought‘, ‘Action‘, and ‘Observation‘ to indicate the sequence of the reasoning steps.
- (3) Please give your ‘Thought‘ first, then the ‘Action‘, and finally the ‘Observation‘, follow the format as shown in the in-context examples below.
- (4) In your ‘Thought‘, you should perform reflection when necessary to ensure the correctness of your reasoning process, such as: "Wait! Maybe I made some mistakes! I need to rethink from scratch", "Alternatively, we can...", "Hold on, let’s try another approach", etc.
- (5) Give your ‘Action‘ in the form of function call, as shown in in-context examples below.
- (6) You should not provide information based on your own knowledge, only use the information provided in the context.

Some example of reflection text: "There is no enough information from the previous steps. I need to plan my query again." "No context found from observation. Let me restart the reasoning process." "Missing information. Let me restructure my query." "Wait! Maybe I made some mistakes! I need to rethink from scratch." "I think I need to take a step back and reconsider my approach." "I need to reevaluate my reasoning process. Let’s start over." "I need to reflect on my reasoning. Let’s try a different approach." {More examples of reflection text. Simplfied for readability}

In-Context Example: {Some in-context examples}

Instruction prompt P for fine-tuning and inference with ReaRAG

Your task is to solve a question answering task. To improve your solving accuracy, please conduct reasoning process interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action are in the form of function, there are two types:

###### Available Functions:

- (1) Search

{

"name": "search", "description": "It can help you find useful information through the internet or local

knowledge base. You can use this tool to access external knowledge.",

"parameters": { "type": "object", "properties": {

"query": {

"description": "what you want to search" }

}, "required": ["query"]

} }

- (2) Finish

{

"name": "finish", "description": "You can use this function to make a conclusion from the reasoning process

and give the final answer. The reasoning process is completed after this ‘finish‘ function is called",

"parameters": { "type": "object", "properties": {

"answer": {

"description": "the final answer" }

}, "required": ["answer"]

}

} Please follow the format strictly.

Answer prompt Pans to derive final answer

<|system|> You are a QA assistant. Always return a short answer. Output ONLY the answer with no extra words.<|user|> You have conducted multiple searches to seek for informations to answer the Question. Searches {Previous search query and search results}

Instruction Answer the Question based on the searches. **Only output the final answer. Do NOT add any explanation, punctuation, or extra words.** Question: {question} Answer: <|assistant|>

Judgement prompt Pjudge for ACCL metrics on all benchmarks except for FanOutQA

You are asked to evaluate the quality of the AI assistant’s answer to user questions as an impartial judge, and your evaluation should take into account factors including correctness (high priority), and comprehensiveness (whether the assistant’s answer covers all points). Read the AI assistant’s answer and compare against the reference answer, and give an overall integer rating in 1, 2, 3 (1 = wrong or irrelevant, 2 = partially correct, 3 = correct and comprehensive) based on the above principles, strictly in the following format:"[[rating]]", e.g. "[[2]]".

[Reference answer] {Ground truth} [Assistant’s answer] {Prediction} Rating:

Judgement prompt Pjudge for ACCL metrics on FanOutQA benchmarks

[BEGIN DATA]

************ [Question]: {Question}

************ [Expert]: {Ground truth}

************ [Submission]: {Prediction}

************ [END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation. The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. First, write out your reasoning in a step-by-step manner to carefully evaluate the factual content. Make sure your conclusion is supported by detailed reasoning. Avoid simply stating the correct choice immediately. After completing your reasoning, output only the final selected letter (A, B, C, D, E, or F). Format your output exactly as follows: wrap the letter inside <answer> and </answer> tags. Important: Do not include any extra words, quotation marks, punctuation, or explanations — only the letter inside the tags.

Here are the available choices:

- (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
- (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
- (C) The submitted answer contains all the same details as the expert answer.
- (D) There is a disagreement between the submitted answer and the expert answer.
- (E) The answers differ , but these differences don’t matter from the perspective of factuality.
- (F) The submitted answer does not answer the question or is otherwise invalid.

Instruction prompts Pablat for data construction to fine-tune ReaRAG w/o reasoning

You are given Question, Ground-truth answer, and Decompositions, your task is to give reasoning process interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action are in the form of function, there are two types:

- (1) search

{

"name": "search", "description": "It can help you find useful information through the internet or local

knowledge base. You can use this tool to access external knowledge.",

"parameters": { "type": "object", "properties": {

"query": {

"description": "what you want to search" }

}, "required": ["query"]

} }

- (2) finish

{

"name": "finish", "description": "You can use this function to make a conclusion from the reasoning process

and give the final answer. The reasoning process is completed after this ‘finish‘ function is called",

"parameters": { "type": "object", "properties": {

"answer": {

"description": "the final answer" }

}, "required": ["answer"]

} }

After ‘finish‘ function being called, the reasoning process is completed, then marked with <|completed|>. Please follow the format strictly.

Here are in-context examples: Question: What was the wettest year in the second largest city in the state where Yuma’s Library District is located?

Ground-truth answer: 1905

Decompositions: [

{"question": "Yuma located in the administrative territorial entity", "answer": "Yuma County", "context": "Some context" # Simplified for readability},

... # Simplified for readability {"question": "Tucson’s wettest year?",

"answer": "1905", "context": "Some context" # Simplified for readability}

] Reasoning process with function call: {Reasoning process example}

### C Case study

|# Case Study 1: Where was the author of Hannibal and Scipio educated at?|
|---|
|Ground truth: Exeter College|
|Search-o1|
|Alright, I’ve got this question: ... Simplified here. So, I’ll do a search for "Hannibal and Scipio book." <|begin_search_query|>Hannibal and Scipio book<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|> Hmm, that didn’t turn up anything. Maybe the book has a different title. Let me think differently. ... Simplified here. Search-o1 reasoning diverges and hence unable to find related information. Only search queries and results are listed below for simplicity.<br><br><|begin_search_query|>Punic Wars book<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|> <|begin_search_query|>Hannibal vs. Scipio book<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|> <|begin_search_query|>Adrian Goldsworthy books<|end_search_query|> <|begin_search_result|>No helpful information found.**Thoughts**After carefully reviewing the three web pages provided, it is evident that none of them ... Simplified here. <|end_search_result|> <|begin_search_query|>Robert F. Pennell education<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|><br><br>... Simplified here. Search-o1 tries many unrelated search queries, demonstrating divergent reasoning. I’ve reached the maximum number of searches. Based on the information I have, I need to make an educated guess. ... Simplified here. Maximum searches reaches<br><br>Search-o1 tries to make a guess when there is not enough information to answer the multi-hop question. Considering that Oxford and Cambridge are traditional centers for classical studies in the UK, I’ll go with Oxford. Therefore, my answer is:<br><br>|Oxford|
|---|
<br><br>|
|ReaRAG-9B|
|Thought 1: So I need to find out where the author of "Hannibal" and "Scipio" was educated. First, I need to know who the author is.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Who is the author of Hannibal and Scipio?'}} Observation 1: The author of "Hannibal and Scipio" is Thomas Nabbes.<br><br>|
|Thought 2: Okay, now that I know the author is Thomas Nabbes, I need to find out where he was educated. Action 2: {'function': 'search', 'parameters': {'query': 'Where was Thomas Nabbes educated?'}} Observation 2: Thomas Nabbes was educated at Exeter College, Oxford.<br><br>|
|Thought 3: Great, so Thomas Nabbes studied at Exeter College in Oxford. I should make sure that this the correct information.<br><br>Action 3: {'function': 'search', 'parameters': {'query': 'Confirm the educational background of Thomas Nabbes'}} Observation 3: Thomas Nabbes was educated at Exeter College, Oxford in 1621. He left the university without taking a degree.<br><br>|
|Thought 4: So, he studied at Exeter College in Oxford, but didn’t finish his degree. That’s interesting. I think this is the information I need. Action 4: {'function': 'finish', 'parameters': {'answer': 'Thomas Nabbes was educated at Exeter College, Oxford.'}}<br><br>|
|Final Answer: Exeter College, Oxford|

- Table 5: We compare the generations of Search-o1 and ReaRAG. During the initial searches, the Reason-inDocuments module of Search-o1 returns “No helpful information found” because it attempts to answer the multi-hop question using information retrieved based on the search query rather than the original question. Hence, the retrieved content lacks the full answer, misleading Search-o1’s reasoning. Ultimately, it reaches its search limits and fails to provide the correct answer. In contrast, ReaRAG demonstrates logical reasoning and correctly answers the query.

|# Case Study 2: What record label did the person who is part of The Bruce Lee Band start?|
|---|
|Ground truth: Asian Man Records|
|Search-o1|
|Alright, so I need to find out what record label the person who is part of The Bruce Lee Band ... Simplified here. So, I’ll do a search for "The Bruce Lee Band." <|begin_search_query|>The Bruce Lee Band<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|><br><br>Hmm, that’s strange. Maybe the band name is different or not well-known. Perhaps I should try searching for "Bruce Lee tribute band" or something like that.<br><br>... Simplified here. Search-o1 tries different searches, but Reason-in-Documents returns “No helpful information” Only search queries and search results are listed here for simplicity<br><br><|begin_search_query|>Bruce Lee tribute band<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|> <|begin_search_query|>Bruce Lee band<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|><br><br>... Simplified here. <|begin_search_query|>The Bruce Lee Band members<|end_search_query|><br><br><|begin_search_result|>Less Than Records, started by a member of Less Than Jake, who is part of The Bruce Lee Band.**Explanation**Based on the information from Web Page 1, The Bruce Lee Band features Less Than Jake as part of its backing band. Less Than Jake is known for starting their own record label, Less Than Records.<br><br>... Simplified here. Search-o1 fabricates the name "Less Than Records" <|begin_search_query|>Less Than Jake record label<|end_search_query|><br><br><|begin_search_result|>No helpful information found.**Explanation**After reviewing the three web pages provided for the search query "Less Than Jake record label," none of them contained specific information about which member of Less Than Jake started Less Than Records. ... Simplified here. <|end_search_result|><br><br>... Simplified here. Less Than Records was started by a member of Less Than Jake, who is also part of The Bruce Lee Band.Therefore, the record label started by the person who is part of The Bruce Lee Band is Less Than Records. Final Answer<br><br>|Less Than Records|
|---|
<br><br>|
|ReaRAG-9B|
|Thought 1: I need to find out which record label the person who is part of The Bruce Lee Band started. First, I should figure out who is part of The Bruce Lee Band.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Who is part of The Bruce Lee Band?'}} Observation 1: Mike Park and his backing band, which has included Less Than Jake and the Rx Bandits, are part of The Bruce Lee Band.<br><br>|
|Thought 2: So, Mike Park is involved with The Bruce Lee Band. I need to find out if he started a record label. Action 2: {'function': 'search', 'parameters': {'query': 'Did Mike Park start a record label?'}} Observation 2: Yes, Mike Park started a record label.<br><br>|
|Thought 3: Great, now I need to find out the name of the record label he started. Action 3: {'function': 'search', 'parameters': {'query': 'What record label did Mike Park start?'}} Observation 3: Mike Park started the record label Asian Man Records.<br><br>|
|Thought 4: So, Asian Man Records is the label Mike Park started. But I should make sure that this is correct and see if there’s any other relevant information.<br><br>Action 4: {'function': 'search', 'parameters': {'query': 'Confirm if Asian Man Records was started by Mike Park.'}} Observation 4: Yes, Asian Man Records was started by Mike Park.<br><br>|
|Thought 5: Alright, I have confirmed that Mike Park started Asian Man Records. Since he is part of The Bruce Lee Band, this should be the answer.<br><br>Action 5: {'function': 'finish', 'parameters': {'answer': 'Asian Man Records'}}|
|Final Answer: Asian Man Records|

- Table 6: We compare the generations of Search-o1 and ReaRAG. Search-o1’s Reason-in-Documents module fabricates non-existent information ("Less Than Records", highlighted in red), compromising factual accuracy and leading to a hallucinated response. In contrast, ReaRAG successfully conduct knowledge-guided reasoning to generate a factual answer.

|# Case Study 3: What is the scientific name of an amphibian that is listed as endangered by the Canadian government (as of 2024), and its only population in Canada occurs on an island which is the southernmost inhabited part of Canada?|
|---|
|Ground truth: Ambystoma texanum|
|R1-Searcher|
|To solve this question, I need to find the scientific name of an amphibian that is listed as endangered by the Canadian government and has its only population in Canada on an island that is the southernmost inhabited part of Canada.<br><br>Step 1: Identify the island that is the southernmost inhabited part of Canada.<br><br>Step 2: Determine which amphibian is listed as endangered by the Canadian government and has its only population on that island.<br><br><br>**Step 1: Identify the island that is the southernmost inhabited part of Canada.** I’ll use a query to find out: <|begin_of_query|> southernmost inhabited part of Canada<|end_of_query|> <|begin_of_documents|> Pelee, Ontario:<br><br>Pelee is the southernmost inhabited part of Canada and is situated on two major migratory bird routes, the Atlantic Flyway and the Mississippi flyway. The<br><br>Extreme points of Canada: The southernmost point is Middle Island, in Lake Erie, Ontario (41°41’N 82°40’W); the southernmost water point lies just south of the island, on the Ontario– Middle Island (Lake Erie):<br><br>Middle Island is a small island, just 18.5 hectares (46 acres) in area. [1] It is the southernmost point of land in Canada, located at 41°41’N, 82°41"W. Extreme points of Canadian provinces: British Columbia, Yukon and Northwest Territories border ... Simplified here. Southern Ontario: Southern Ontario is a primary region of the Canadian province of Ontario. ... Simplified here. <|end_of_documents|><br><br>The keyword "southernmost" appears in multiple documents, but only Pelee, Ontario is correct as the southernmost inhabited part of Canada. R1-Searcher fails to identify it accurately.<br><br>From the search results, I can see that Middle Island is the southernmost inhabited part of Canada.<br><br>**Step 2: Determine which amphibian is listed as endangered by the Canadian government and has its only population on Middle Island.** I’ll use a query to find out: <|begin_of_query|> amphibian listed as endangered by the Canadian government with its only population on Middle Island, Ontario<|end_of_query|> Wrong query due to earlier mistakes. Middle Island is not the correct place. <|begin_of_documents|> List of amphibians of Canada: List of amphibians of Canada ... Simplified here. List of Wildlife Species at Risk (Canada): As of November 2024, ... Simplified here. List of endangered amphibians: Of all evaluated amphibian species, ... Simplified here. [PDF] Midland Painted Turtle - Species at risk public registry: Painted Turtle is recognized as a species of Least Concern ... Simplified here. American burying beetle (Nicrophorus americanus): COSEWIC : The American Burying Beetle ... Simplified here. .<|end_of_documents|><br><br><br>From the search results, I can see that there is no specific amphibian listed as endangered by the Canadian government with its only population on Middle Island, Ontario. However, I will focus on amphibians that are endangered and have populations in Canada.</think><br><br><answer> Midland Painted Turtle </answer><br><br>|
|ReSearch|
|<think> To find the scientific name of the amphibian that is listed as endangered by the Canadian government and is only found on the southernmost inhabited island of Canada, I need to first identify the island and then find the amphibian species that is endangered there. Let’s start by identifying the southernmost inhabited island of Canada.</think><br><br><search> search query: What is the southernmost inhabited island of Canada?</search> <result><br><br>|

|# Case Study 3 (continued)|
|---|
|Extreme points of Canada:<br><br>The southernmost point is Middle Island, in Lake Erie, Ontario (41°41’N 82°40’W); the southernmost water point lies just south of the island, on the Ontario<br><br>Middle Island (Lake Erie): Middle Island is a small island, just 18.5 hectares (46 acres) in area. [1] It is the southernmost point of land in Canada, located at 41°41’N, 82°41"W. Pelee, Ontario:<br><br>At 42 km2 (16 sq mi), Pelee Island is the largest island in Lake Erie and the southernmost populated point in Canada. An Ontario Historical Plaque was erected<br><br>Ontario: Ontario is the southernmost province of Canada. ... Simplified here. Newfoundland (island): Newfoundland is a large island within the Canadian province ... Simplified here.<br><br>Similar to R1-Searcher, ReSearch mistakenly identified Middle Island as the southernmost inhabited island leading to a subsequent incorrect query.<br><br></result><think> From the information provided, the southernmost inhabited island of Canada is Middle Island in Lake Erie, Ontario. Now, I need to find out which amphibian species is listed as endangered on this island.</think> <search> search query: Which amphibian is endangered on Middle Island, Ontario?</search> <result> List of amphibians of Canada:<br><br>List of amphibians of Canada ; EX - Extinct, EW - Extinct in the Wild ; CR - Critically Endangered, EN - Endangered, VU - Vulnerable ; NT - Near Threatened, LC List of endangered amphibians:<br><br>As of September 2021, the International Union for Conservation of Nature (IUCN) lists 1085 endangered amphibian species.<br><br>List of Wildlife Species at Risk (Canada): Threatened amphibians · Ambystoma jeffersonianum — Jefferson salamander · Bufo fowleri — Fowler’s toad · Desmognathus ochrophaeus — Allegheny Mountain dusky<br><br>... Simplified here. No related information about endangered species on Middle Island.<br><br></result><think> The information provided does not directly mention the amphibian species that is endangered on Middle Island. I need to narrow down the search to amphibians that are specifically listed as endangered in Canada and are found on Middle Island.</think><br><br><search> search query: Endangered amphibian on Middle Island, Ontario, Canada</search> <result> List of amphibians of Canada:<br><br>This is a list of amphibians of Canada. Conservation status - IUCN Red List of Threatened Species: EX - Extinct, EW Extinct in the Wild: CR - Critically<br><br>List of endangered amphibians: As of September 2021, the International Union for Conservation of Nature (IUCN) lists 1085 endangered amphibian species. Middle Island (Lake Erie):<br><br>Middle Island is a small island, just 18.5 hectares (46 acres) in area. [1] It is the southernmost point of land in Canada, located at 41°41’N, 82°41"W.<br><br>... Simplified here. No related information about endangered species on Middle Island.<br><br></result><think> After reviewing the information, I realize that the specific amphibian species that is listed as endangered on Middle Island, Ontario, Canada, is the Jefferson salamander (Ambystoma jeffersonianum). This amphibian is the only one that fits the criteria of being endangered and only found on this specific island.</think><br><br><think> Now that I have identified the amphibian species, I can provide the scientific name.</think> <answer><br><br>|Ambystoma jeffersonianum|
|---|
<br><br></answer><br><br>|
|ReaRAG-9B|
|Thought 1: So I need to find the scientific name of an endangered amphibian in Canada, and its only population in Canada is on an island that’s the southernmost inhabited part of Canada. Okay, first things first, I need to know what the southernmost inhabited island in Canada is.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'What is the southernmost inhabited island in Canada?'}} Observation 1: The southernmost inhabited island in Canada is Pelee Island, Ontario.<br><br>|
|Thought 2: Alright, so the island is Pelee Island in Ontario. Now, I need to find an endangered amphibian that’s only found there in Canada. Maybe there are some lists of endangered species in Canada that I can look at.|

|# Case Study 3 (continued)|
|---|
|Action 2: {'function': 'search', 'parameters': {'query': 'List of endangered amphibians in Canada'}} Observation 2: I am sorry, but I cannot provide a specific list of endangered amphibians in Canada based on the given context. The context provided does not include a list of endangered amphibians in Canada.|
|Thought 3: Hmm, it seems like I need to find out which amphibians are endangered in Canada and see if any of them are only found on Pelee Island. Maybe I should look for information on endangered species in Ontario, since Pelee Island is in Ontario.<br><br>Action 3: {'function': 'search', 'parameters': {'query': 'Endangered amphibians in Ontario'}} Observation 3: The endangered amphibians in Ontario mentioned in the context are: 1. Jefferson salamander (Ambystoma jeffersonianum) ... 2. Northern dusky salamander (Desmognathus fuscus) ...<br><br>Both species are actually not on Peele Island. ReaRAG later verifies this information.|
|Thought 4: So, there are a couple of endangered amphibians in Ontario, but I need to find out if any of them are only found on Pelee Island. Maybe I should look into the distribution of these species.<br><br>Action 4: {'function': 'search', 'parameters': {'query': 'Distribution of Jefferson salamander in Ontario'}} Observation 4: The distribution of the Jefferson salamander in Ontario is primarily associated with the Niagara Escarpment and Carolinian forest regions.<br><br>|
|Thought 5: Pelee Island is part of the Lake Erie Islands, which are different from the Niagara Escarpment and Carolinian forest regions. So, perhaps the Jefferson salamander isn’t found there. Maybe I should check the distribution of the Northern dusky salamander.<br><br>Action 5: {'function': 'search', 'parameters': {'query': 'Distribution of Northern dusky salamander in Ontario'}} Observation 5: The Northern dusky salamander (Desmognathus fuscus) is restricted to a small area in Ontario. Specifically, it can be found in the Niagara Gorge.<br><br>|
|Thought 6: Again, Niagara Gorge is different from Pelee Island. Maybe there’s another endangered amphibian in Ontario that’s only found on Pelee Island. Perhaps I need to look for amphibians that are specific to Pelee Island.<br><br>Now, ReaRAG performs a more specific search. Action 6: {'function': 'search', 'parameters': {'query': 'Endangered amphibians specific to Pelee Island, Ontario'}} Observation 6: Small-mouth salamander.<br><br>|
|Thought 7: So, the small-mouth salamander is found on Pelee Island, but I need to confirm if it’s endangered in Canada. Action 7: {'function': 'search', 'parameters': {'query': 'Is the small-mouth salamander endangered in Canada?'}} Observation 7: Yes, the small-mouth salamander is listed as endangered in Canada.<br><br>|
|Thought 8: Great, so now I have the scientific name, Ambystoma maculatum, which is the small-mouth salamander. But wait, I need to make sure that it’s only found on Pelee Island in Canada.<br><br>Action 8: {'function': 'search', 'parameters': {'query': 'Distribution of small-mouth salamander in Canada'}} Observation 8: The distribution of the small-mouth salamander (Ambystoma texanum) in Canada is restricted solely to Pelee Island. The entire Canadian range of this species is only on Pelee Island.<br><br>|
|Thought 9: Wait a minute, in my earlier steps, I thought it was Ambystoma maculatum, but now the search is showing Ambystoma texanum. There might be some confusion here.<br><br>Despite earlier mistakes, ReaRAG managed to resolve the confusion by taking correct search action Action 9: {'function': 'search', 'parameters': {'query': 'Scientific name of small-mouth salamander'}} Observation 9: The scientific name of the small-mouth salamander is Ambystoma texanum.<br><br>|
|Thought 10: Okay, so it’s Ambystoma texanum, not Ambystoma maculatum. I must have misremembered earlier. So, to confirm, Ambystoma texanum is the small-mouth salamander, and it’s endangered in Canada, and its only population in Canada is on Pelee Island.<br><br>Action 10: {'function': 'finish', 'parameters': {'answer': 'Ambystoma texanum.'}}<br><br>|
|Final Answer: Ambystoma texanum|

- Table 7: This case study illustrates the poor generalization of R1-Searcher on FRAMES and FanOutQA. Both R1Searcher and ReSearch fail to extract the correct information from noisy retrieved documents, mistakenly identifying “Middle Island” as Canada’s southernmost inhabited area instead of “Pelee, Ontario”, leading to errors in subsequent steps. In contrast, despite early mistakes, ReaRAG successfully realigns its reasoning path, demonstrating greater robustness.

|# Case Study 4: In what year was each member of Blackpink born?|
|---|
|Ground truth: Jisoo - 1995, Jennie - 1996, Rosé - 1997, Lisa - 1997|
|R1-Searcher|
|To solve this question, I need to find out the birthdates of each member of Blackpink.<br><br>Step 1: Identify the members of Blackpink. I’ll use a query to search for information about the members of Blackpink: <|begin_of_query|> Blackpink members<|end_of_query|> <|begin_of_documents|> Blackpink: The group is composed of four members: Jisoo, Jennie, Rosé, and Lisa. ... Simplified here. <|end_of_documents|> From the search results, I can see that the members of Blackpink are Jisoo, Jennie, Rosé, and Lisa.<br><br>Step 2: Find the birthdates of each member. I’ll use a query to search for the birthdates of Jisoo, Jennie, Rosé, and Lisa: <|begin_of_query|> Jisoo birthdate, Jennie birthdate, Rosé birthdate, Lisa birthdate<|end_of_query|> <|begin_of_documents|> Blackpink: Blackpink is a South Korean girl group formed by YG Entertainment. ... Simplified here. Jisoo: Jisoo ; Born. Kim Ji-soo. (1995-01-03) January 3, 1995 (age 30). ... Simplified here. . Rosé (singer): Roseanne Park (born 11 February 1997), known mononymously as Rosé, is a New Zealand ... Simplified here. Jennie (singer): Jennie ; Born. Jennie Kim. (1996-01-16) January 16, 1996 ... Simplified here. Blackpink - Simple English Wikipedia, the free encyclopedia:<br><br><br>The group consists of members Jisoo, Jennie, Rosé, and Lisa. They are the highest-charting female K-pop act on both Billboard Hot 100 and Billboard 200.<br><br><|end_of_documents|> From the search results, I can see the following birthdates:<br><br>- Jisoo: January 3, 1995<br><br>- Jennie: January 16, 1996<br><br>- Rosé: February 11, 1997<br><br>- Lisa: Unknown (only age mentioned) Therefore, the year each member of Blackpink was born is:<br><br>- Jisoo: 1995<br><br>- Jennie: 1996<br><br>- Rosé: 1997<br><br>- Lisa: Unknown </think> Although Lisa’s birthdate is missing, R1-Searcher proceeds to generate a final answer.<br><br><br><answer> 1995, 1996, 1997, Unknown </answer><br><br>|
|ReSearch|
|<think> To find the birth years of each member of Blackpink, I will first search for the individual members of the group and then look for their birth years. I will use the Wikipedia search tool to find this information.</think> <search> search query: Members of Blackpink</search> <result> Blackpink:<br><br>The group is composed of four members: Jisoo, Jennie, Rosé, and Lisa. Regarded by various publications as the "biggest girl group in the world", they are<br><br>... Simplified here. </result><think> Now that I have the names of the members, I will search for their birth years.</think> <search> search query: Jisoo birth year</search> <result> Jisoo:<br><br>Jisoo ; Born. Kim Ji-soo. (1995-01-03) January 3, 1995 (age 30). Gunpo, South Korea ; Education, School of Performing Arts Seoul ; Occupations. Singer; actress.<br><br>|

|# Case Study 4 (continued)|
|---|
|Jisoo - Simple English Wikipedia, the free encyclopedia:<br><br>Jisoo ; Born, (1995-01-03) January 3, 1995 (age 30). South Korea ; Occupations. Singer; actress; model ; Origin, Seoul, South Korea ; Genres. K-pop · dance-pop<br><br>Ji Soo: Kim Ji-soo (born 30 March 1993), professionally known as Ji Soo, is a South Korean actor currently based in the Philippines. Kim Ji-soo (actress, born 1972):<br><br>Kim Ji-soo is a South Korean actress. Kim Ji-soo. Kim Ji-soo at the 5th World Ceramic Biennale press conference in Korea (2009). Born. Yang Sung-yoon.<br><br>Kim Ji-soo (singer, born 1990): Kim Ji-soo (singer, born 1990) ; (1990-02-05) February 5, 1990 (age 35) South Korea · Pop; folk · Singer-songwriter; musician. </result><think> I have found the birth year of Jisoo. Now I will search for the birth years of the other members.</think> <search> search query: Jennie birth year</search> <result> Jennie (singer):<br><br>Jennie ; Born. Jennie Kim. (1996-01-16) January 16, 1996 (age 29). Seongnam, Gyeonggi, South Korea ; Other names, Jennie Ruby Jane ; Occupations. Singer; rapper Jennie (singer) - Simple English Wikipedia, the free encyclopedia:<br><br>Jennie Kim (born January 16, 1996), known as Jennie is a South Korean singer and rapper. She is a member of the South Korean girl<br><br>Jennie Baranczyk: Jennie Lillis Baranczyk is an American basketball coach who is the current head coach of the University of Oklahoma women’s basketball team. Jennie Jennie:<br><br>Jennie (film), a 1940 American drama film · Jennie (musical), a 1963 Broadway production · Jennie (novel), a 1994 science fiction thriller by Douglas Preston Jennie Garth:<br><br>Jennie Garth ; Garth in 2023 ; Born. Jennifer Eve Garth. (1972-04-03) April 3, 1972 (age 53). Urbana, Illinois, U.S. ; Occupation(s), Actress, fashion designer.<br><br></result><think> I have found the birth year of Jennie. Now I will search for the birth years of the other members.</think><br><br><search> search query: Rosé birth year</search> <result> Rosé (singer):<br><br>Rosé. MBE ; Born. Roseanne Park. (1997-02-11) 11 February 1997 (age 28). Auckland, New Zealand ; Other names, Park Chae-young ; Citizenship. South Korea · New<br><br>Rosé (singer) - Simple English Wikipedia, the free encyclopedia: Rosé was born in New Zealand on 11 February 1997. At the age of seven, she moved to Australia. In 2012, she passed a YG Entertainment audition at the age of Rose (French singer):<br><br>Keren Meloul (born May 24, 1978), better known as Rose, is a French singer, songwriter, author and composer. Rose. Rose performing in 2008.<br><br>Rosé discography: Rosé began her career as a member of girl group Blackpink in August 2016. She released her debut solo single album R through YG Entertainment and Interscope Rose Falcon:<br><br>Rose Falcon (born May 2, 1984) is an American singer and songwriter. Falcon’s songs have been recorded by Faith Hill, Lady A, Day of Fire, Eric Paslay,<br><br></result><think> I have found the birth year of Rosé. Now I will search for the birth year of the last member.</think> <search> search query: Lisa birth year</search> <result> Lisa (rapper):<br><br>Lisa in 2024 ; Born. Pranpriya Manobal. (1997-03-27) March 27, 1997 (age 28). Buriram, Thailand ; Occupations. Rapper; singer; dancer; actress; songwriter. Lisa:<br><br>|

###### Continued on next page

|# Case Study 4 (continued)|
|---|
|Lisa (Japanese musician, born 1974), stylized "LISA"; Lisa, stagename of Japanese singer Lisa Komine (born 1978); Lisa (South Korean singer) (born 1980); Lisa Lisa Ajax:<br><br>Birth name, Lisa Kristina Ajax ; Born, (1998-08-13) 13 August 1998 (age 26) Järfälla, Sweden ; Genres, Pop ; Years active, 2012–present ; Labels, Capitol Records<br><br>Lisa Collier Cool: Lisa Collier Cool (born July 6, 1952) is an American journalist and author who writes on health-related topics. Lisa Collier Cool. Born, July 6, 1952. Lisa (Japanese musician, born 1987):<br><br>Born. Risa Oribe. (1987-06-24) June 24, 1987 (age 37). Seki, Gifu, Japan ; Occupations. Singer; songwriter ; Years active, 2005–present ; Spouse. Tatsuhisa Suzuki.<br><br></result><think> I have found the birth year of Lisa. Now I can compile the birth years of all the members of Blackpink.</think> <think> Having found the birth years of each member, I can now provide the answer.</think><br><br>ReSearch fails to reason over previous steps and provide an accurate answer due to noisy information. <answer><br><br>|2018|
|---|
<br><br></answer><br><br>|
|ReaRAG-9B|
|Thought 1: I need to find out the birth years of all four members of Blackpink. I should probably look up their names individually to get their birthdates.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Blackpink members and their birth years'}} Observation 1: Obs 1: Blackpink members and their birth years are as follows: - Jisoo: 1995<br><br>- Jennie: 1996<br><br>- Rosé: 1997<br><br>- Lisa: 1997<br>|
|Thought 2: Wait, I should verify this information because sometimes birthdates can be confusing, especially if there are different sources with different dates.<br><br>Action 2: {'function': 'search', 'parameters': {'query': 'Confirm birth years of Blackpink members'}} Observation 2: The birth years of Black Blackpink members are as follows: - Jisoo: 1995<br><br>- Jennie: 1996<br><br>- Rosé: The provided in the context<br><br>- Lisa: 1997<br><br><br>|
|Thought 3: Hmm, it seems like there’s a gap in the information for Rosé. I need to find her birth year specifically. Action 3: {'function': 'search', 'parameters': {'query': 'Rosé Blackpink birth year'}} Observation 3: 1997<br><br>|
|Thought 4: Okay, so Rosé was born in 1997. Now, I should make sure that all the birth years are correct and consistent across different sources.<br><br>Action 4: {'function': 'search', 'parameters': {'query': 'Cross-verify birth years of Blackpink members'}} Observation 4: The birth years of the Blackpink members are as follows: - Jisoo: January 3, 1995<br><br>- Jennie: January 16, 1996<br><br>- Rosé: February 11, 1997<br><br>- Lisa: March 27, 1997<br>|
|Thought 5: Great, now I have the birth years along with the birthdates for each member. This should be sufficient to answer the question. Action 5: {'function': 'finish', 'parameters': {'answer': 'Jisoo was born in 1995, Jennie in 1996, Rosé in 1997, and Lisa in 1997.'}}<br><br>|
|Final Answer: Jisoo: 1995, Jennie: 1996, Rosé: 1997, Lisa: 1997|

- Table 8: This case study illustrates the poor generalization of R1-Searcher and ReSearch on the FRAMES and FanOutQA benchmarks. Specifically, R1-Searcher fails to resolve missing information about Lisa’s birthdate and does not take additional steps to complete the reasoning process. ReSearch, on the other hand, constructs a noisy reasoning chain, leading to an incorrect answer. In contrast, despite also facing missing information (Rosé’s birthdate), ReaRAG performs additional verification and successfully recovers the correct answer, demonstrating greater robustness.

|# Case Study 5: In what county is William W. Blair’s birthplace located?|
|---|
|Ground truth: Orleans County|
|Search-o1|
|... Simplified here. Maybe I can search for "Holley, New York county" to find out which county it’s part of. <|begin_search_query|>Holley, New York county<|end_search_query|> <|begin_search_result|>Holley is located in Orleans County, New York.<|end_search_result|><br><br>... Simplified here. But wait, let me double-check to make sure that Holley is indeed in Orleans County. Sometimes place names can be ambiguous, and there might be multiple Holleys in different counties or states. Let me search for "Holley, New York county" again to confirm. <|begin_search_query|>Holley, New York county<|end_search_query|> <|begin_search_result|>You have searched this query. Please refer to previous results.<|end_search_result|> Alright, since the previous search already indicated that Holley is in Orleans County, New York, and no other information contradicts that, I can confidently say that William W. Blair was born in Orleans County. However, to be thorough, perhaps I should verify if there are any other places named Holley in New York or elsewhere that might be confused with the one where William W. Blair was born. Let me search for "Holley, New York" to see if there’s only one such place. <|begin_search_query|>Holley, New York<|end_search_query|> <|begin_search_result|>Holley is a village ... Simplified here. <|end_search_result|><br><br>But just to cover all bases, maybe there’s another Holley in New York state that I should consider. Let me search for "New York state places named Holley" to see if there are multiple locations with that name.<br><br><|begin_search_query|>New York state places named Holley<|end_search_query|> <|begin_search_result|>Holley is a village ... Simplified here. <|end_search_result|><br><br>Search-o1 continues to execute excessive and unnecessary searches. Only search queries are listed here for simplicity. <|begin_search_query|>William W. Blair historian<|end_search_query|> <|begin_search_query|>William W. Blair Latter Day Saints<|end_search_query|> <|begin_search_query|>William W. Blair birth certificate Orleans County<|end_search_query|> <|begin_search_query|>William W. Blair biography<|end_search_query|><br><br>Perfect, so this biography confirms that he was born in Holley, Orleans County, New York. This aligns with all the previous information I’ve found.<br><br>Therefore, I can confidently say that William W. Blair’s birthplace is located in Orleans County. ... Simplified here. Final Answer<br><br>|Orleans County|
|---|
<br><br>|
|ReaRAG-9B|
|Thought 1: I need to find out in which county William W. Blair was born. First, I should figure out where he was born, meaning his birthplace city or town.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Where was William W. Blair born? '}} Observation 1: William W. Blair was born in Holley, New York.<br><br>|
|Thought 2: Okay, so now I know that his birthplace is Holley, New York. Next, I need to determine which county Holley is located in.<br><br>Action 2: {'function': 'search', 'parameters': {'query': 'Which county is Holley, New York in?'}} Observation 2: Holley, New York is in Orleans County.<br><br>|
|Thought 3: So, if Holley is in Orleans County, then that should be the answer. But I should double-check if there’s any ambiguity or if there are multiple places named Holley in New York.<br><br>Action 3: {'function': 'search', 'parameters': {'query': 'Are there multiple Holleys in New York State?'}}<br><br>Observation 3: Based on the provided context, there is only one Holley mentioned, which is in Orleans County, New York. The context does not provide information about other Holleys in New York State. Therefore, based on the given context, I cannot confirm whether there are multiple Holleys in New York State.<br><br>|
|Thought 4: Given that, and considering the information I have, I think it’s safe to conclude that William W. Blair’s birthplace, Holley, is in Orleans County, New York.<br><br>Action 4: {'function': 'finish', 'parameters': {'answer': 'Orleans County'}}<br><br>|
|Final Answer: Orleans County|

- Table 9: We compare the generations of Search-o1 and ReaRAG. Words highlighted in red indicate instances where Search-o1 overthinks and performs excessive verification. Words highlighted in green show that the correct answer was already found in earlier steps. ReaRAG finds the correct answer in just two searches, whereas Search-o1 requires eleven, generating overly redundant queries that introduce unnecessary computations to multi-hop QA.

|# Case Study 6: Where is Ulrich Walter’s employer headquartered?|
|---|
|Ground truth: Cologne|
|Search-o1|
|Alright, I have this question: "Where is Ulrich Walter’s employer ... Simplified here. <|begin_search_query|>Who is Ulrich Walter?<|end_search_query|><br><br><|begin_search_result|>Ulrich Walter is a German astronaut who was selected in 1987 to join the German astronaut team and trained at the German Aerospace Center (DLR). His employer, the German Aerospace Center, is headquartered in Cologne, Germany.<|end_search_result|><br><br>Hmm, according to this search result, Ulrich Walter is a German astronaut who was selected in 1987 and trained at the German Aerospace Center (DLR). It also says that his employer, the DLR, is headquartered in Cologne, Germany.<br><br>... Simplified here. <|begin_search_query|>Headquarters of German Aerospace Center<|end_search_query|><br><br><|begin_search_result|>The German Aerospace Center (DLR) is headquartered in Cologne, Germany.<|end_search_result|><br><br>So, according to this, the DLR is headquartered in Cologne, Germany. But just to be thorough, let me see if there is any other information about Ulrich Walter’s employer. <|begin_search_query|>Ulrich Walter workplace<|end_search_query|> <|begin_search_result|>Ulrich Walter’s employer is the German Aerospace Center (DLR), headquartered in Cologne, Germany.<|end_search_result|><br><br>... Simplified here. I should probably confirm if Ulrich Walter is still active or if he has retired. ... Simplified here. <|begin_search_query|>Ulrich Walter retirement<|end_search_query|> <|begin_search_result|>No helpful information found.<|end_search_result|> ... Simplified here. Alternatively, perhaps he has taken up an academic position at a university. <|begin_search_query|>Ulrich Walter university affiliation<|end_search_query|> <|begin_search_result|>Ulrich Walter’s employer, the German Aerospace Center (DLR), is headquartered in Cologne, Germany.<|end_search_result|><br><br>... Simplified here. Maybe I should look for his professional profile or LinkedIn page to see his current employer. <|begin_search_query|>Ulrich Walter LinkedIn<|end_search_query|><br><br>... Simplified here. the answer to the question "Where is Ulrich Walter’s employer headquartered?" is Cologne, Germany. Final Answer<br><br>|Cologne, Germany|
|---|
<br><br>|
|ReaRAG-9B|
|Thought 1: I need to find out where Ulrich Walter’s employer is headquartered. First, I should figure out who Ulrich Walter is and where he works.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Who is Ulrich Walter and where does he work?'}} Observation 1: Ulrich Walter is a German astronaut who worked at the German Aerospace Center.<br><br>|
|Thought 2: So, Ulrich Walter works at the German Aerospace Center. Now, I need to find out where this center is headquartered.<br><br>Action 2: {'function': 'search', 'parameters': {'query': 'Where is the German Aerospace Center headquartered?'}} Observation 2: The German Aerospace Center (DLR) is headquartered in Cologne, Germany.<br><br>|
|Thought 3: Alright, so if DLR is headquartered in Cologne, Germany, then that should be the answer. But maybe I should double-check if Ulrich Walter works there still or if there’s any update. Action 3: {'function': 'search', 'parameters': {'query': 'Is Ulrich Walter still working at DLR?'}}<br><br>Observation 3: I cannot provide real-time information or updates on whether Ulrich Walter is still working at the German Aerospace Center (DLR) since my training data only goes up until 2023. To get the most current information on Ulrich Walter’s employment status, you would need to check the latest sources or contact the German Aerospace Center directly.<br><br>|
|Thought 4: Given that, I should probably proceed with the information I have, which is that Ulrich Walter works at DLR, which is headquartered in Cologne, Germany. Action 4: {'function': 'finish', 'parameters': {'answer': 'Cologne, Germany'}}<br><br>|
|Final Answer: Cologne, Germany|

- Table 10: We compare the generations of Search-o1 and ReaRAG. Words highlighted in red indicate instances where Search-o1 overthinks and performs excessive verification. Words highlighted in green show that the correct answer was already found in earlier steps. As shown above, ReaRAG arrives at the correct answer in just three searches, while Search-o1 requires eight, producing excessive queries that are redundant in multi-hop QA.

|# Case Study 7: Who is the grandmother of Philippe, Duke of Orléans?|
|---|
|Ground truth: Marie de’ Medici|
|ReaRAG-9B|
|Thought 1: So I need to find out who the grandmother of Philippe, Duke of Orléans, is. I don’t know much about French royalty, so I’ll need to look this up.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Who is the grandmother of Philippe, Duke of Orléans?'}} Observation 1: The grandmother of Philippe, Duke of Orléans (Philippe I) is Anne of Austria. Wrong answer!<br><br>|
|Thought 2: Wait, let me make sure about this. I have found that his grandmother is Anne of Austria, but I should verify this information from multiple sources to ensure accuracy.<br><br>Action 2: {'function': 'search', 'parameters': {'query': 'Philippe, Duke of Orléans family tree'}}<br><br>Observation 2: Philippe, Duke of Orléans, was the younger son of Louis XIII of France and Anne of Austria. His older brother was Louis XIV, famously known as the "Sun King." Philippe became Duke of Orléans upon the death of his uncle Gaston in 1660. In 1661, he also received the dukedoms of Valois and Chartres.|
|Thought 3: So, according to this, his grandmother is Anne of Austria. But let me see if there’s any confusion here. Philippe’s parents were Louis XIII and Anne of Austria, so she would be his mother, not grandmother. Wait, that doesn’t make sense. Maybe I need to clarify the generational relationships. ReaRAG notices the mistakes.<br><br>Action 3: {'function': 'search', 'parameters': {'query': 'Genealogy of the House of Orléans '}}<br><br>Observation 3: The House of Orléans is a cadet branch of the House of Bourbon, which is one of the most prominent royal houses in European history. The Orléans branch originated with Philippe I, Duke of Orléans, who was the younger son of Louis XIII of France and Anne of Austria. Philippe I was the brother of Louis XIV, the "Sun King." The genealogy of the House of Orléans can be traced through various descendants of Philippe I, including: 1. Philippe II, Duke of Orléans, who was the father of Louise Élisabeth d’Orléans mentioned in Document 351. Philippe II was also known as Philippe I’s grandson, as he was the son of Philippe I’s son, Louis XIV. 2. Louis Philippe, Duke of Orléans, who became King of the French after the July Revolution of 1830. He was the father of François d’Orléans, Prince of Joinville, mentioned in Document 393. These are just a few examples of the descendants of Philippe I, Duke of Orléans, and the House of Orléans continues to have living members today.<br><br>|
|Thought 4: Hmm, I think I’m getting confused here. Philippe I, Duke of Orléans, was the son of Louis XIII and Anne of Austria. So, Anne of Austria would be his mother, not his grandmother. His grandmother would be the mother of his parents, which would be Louis XIII’s mother. Let me check who that was. ReaRAG realigns its reasoning direction.<br><br>Action 4: {'function': 'search', 'parameters': {'query': 'Who was the mother of Louis XIII of France? '}} Observation 4: The mother of Louis XIII of France was Marie de’ Medici.<br><br>|
|Thought 5: So, if Marie de’ Medici was the mother of Louis XIII, and Anne of Austria was his wife and the mother of Philippe I, then Anne of Austria would be Philippe I’s mother, not grandmother. So, his grandmother would be Marie de’ Medici. That makes more sense. But to be thorough, I should confirm the relationships again. Louis XIII married Anne of Austria, and they had children, including Philippe I and Louis XIV. So, Anne of Austria is Philippe I’s mother, and Marie de’ Medici is his grandmother. I think that’s correct, but I’ll double-check with another search.<br><br>Action 5: {'function': 'search', 'parameters': {'query': 'Philippe, Duke of Orléans family tree '}}<br><br>Observation 5: Philippe, Duke of Orléans (1640-1701) was the younger son of Louis XIII of France and Anne of Austria. His older brother was Louis XIV, the famous "Sun King." Philippe was styled Duke of Anjou from birth and became Duke of Orléans upon the death of his uncle Gaston in 1660. He also received the dukedoms of Valois and Chartres in 1661. Philippe had several children, including Louise Élisabeth d’Orléans, who was one of his seven daughters. Louise Élisabeth was known as Élisabeth and was a "princesse du sang" as a member of the reigning House of Bourbon.|
|Thought 6: Okay, so this confirms that Anne of Austria was Philippe I’s mother. Therefore, his grandmother would be Marie de’ Medici, who was Louis XIII’s mother. I think that’s the correct answer. But just to be absolutely sure, let’s look up the genealogy again. Louis XIII was the son of Henry IV and Marie de’ Medici, so Marie de’ Medici would be his mother, and Anne of Austria would be his wife and Philippe I’s mother. I think that’s consistent across all the information I’ve found. So, the grandmother of Philippe, Duke of Orléans, is Marie de’ Medici. I’ll go ahead and provide that as the answer.<br><br>Action 6: {'function': 'finish', 'parameters': {'answer': 'Marie de’ Medici'}}|
|Final Answer: Marie de’ Medici|

- Table 11: In the first reasoning step, ReaRAG makes an incorrect deduction, mistakenly identifying Anne of Austria as Philippe’s grandmother instead of his mother. However, it quickly recognizes the error, then verifies its previous reasoning steps, and recalibrates its logic. By refining its approach, ReaRAG ultimately arrives at the correct answer, demonstrating its adaptive reasoning ability.

|# Case Study 8: who is the younger brother of The episode guest stars of The Hard Easy|
|---|
|Ground truth: Bill Murray|
|ReaRAG-9B|
|Thought 1: I need to find out who the younger brother is of the episode guest stars of "The Hard Easy." First, I should figure out what "The Hard Easy" is. It sounds like a TV show or a movie.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'What is The Hard Easy?'}}<br><br>Observation 1: The Hard Easy is a 2006 American action crime drama film directed by Ari Ryan and written by Jon Lindstrom and Tom Schanley. It stars Henry Thomas, Vera Farmiga, David Boreanaz, Bruce Dern, Peter Weller, Gary Busey, and Nick Lachey. The film was released straight-to-DVD in the United States on May 22, 2007.<br><br>|
|Thought 2: So, "The Hard Easy" is a 2006 film with several guest stars. I need to identify the guest stars and then find out who the younger brother of one of them is. But the question seems a bit unclear. Maybe it’s asking for the younger brother of one of the guest stars in a specific episode, but since it’s a film, not a TV series, perhaps there are multiple episodes.<br><br>Action 2: {'function': 'search', 'parameters': {'query': 'Are there multiple episodes of The Hard Easy?'}} Observation 2: No, there is only one episode of "The Hard Easy" as part of the "Adventure Time" series.<br><br>|
|Thought 3: Wait, now I’m getting confused. Earlier, I thought "The Hard Easy" was a 2006 film, but now it’s saying it’s an episode of "Adventure Time." Maybe there’s some confusion here. I need to clarify what "The Hard Easy" refers to.<br><br>ReaRAG actively performs searches to resolve the confusion. Action 3: {'function': 'search', 'parameters': {'query': 'Is The Hard Easy a film or an episode? '}} Observation 3: The Hard Easy is both a film and an episode. The film version was released straight-to-DVD in 2007, while the episode is part of the animated television series "Adventure Time" and aired on Cartoon Network in 2012.<br><br>|
|Thought 4: Okay, so there’s a 2006 film called "The Hard Easy" and an episode with the same name in the "Adventure Time" series. The question might be referring to the episode in "Adventure Time." I need to find out the guest stars of that episode and then find out who the younger brother of one of them is.<br><br>Action 4: {'function': 'search', 'parameters': {'query': 'Who are the guest stars in the Adventure Time episode "The Hard Easy"? '}} Observation 4: The guest stars in the Adventure Time episode "The Hard Easy" are Brian Doyle-Murray as Prince Huge and Jonathan Katz as the Mudscamp elder.<br><br>|
|Thought 5: So, the guest stars are Brian Doyle-Murray and Jonathan Katz. Now, I need to find out who the younger brother of one of them is. But I don’t know their family relationships off the top of my head. I’ll need to look that up.<br><br>Action 5: {'function': 'search', 'parameters': {'query': 'Who is the younger brother of Brian Doyle-Murray? '}} Observation 5: The younger brother of Brian Doyle-Murray is Bill Murray.<br><br>|
|Thought 6: Wait, that can’t be right. Bill Murray is a famous actor, but I don’t think he’s the younger brother of Brian Doyle-Murray. Maybe I need to check their relationship again.<br><br>ReaRAG encounters conflicts with its parametric knowledge. Action 6: {'function': 'search', 'parameters': {'query': 'Are Bill Murray and Brian Doyle-Murray brothers? '}} Observation 6: Yes, Bill Murray and Brian Doyle-Murray are brothers.<br><br>|
|Thought 7: So, if the question is asking for the younger brother of one of the guest stars in the Adventure Time episode "The Hard Easy," and the guest stars are Brian Doyle-Murray and Jonathan Katz, and the younger brother of Brian Doyle-Murray is Bill Murray, then the answer should be Bill Murray. Action 7: {'function': 'finish', 'parameters': {'answer': 'Bill Murray'}}<br><br>|
|Final Answer: Bill Murray|

- Table 12: During the first three iterations, ReaRAG attempts to resolve ambiguity in the original multi-hop question, actively conducting searches to eliminate confusion. Additionally, at the sixth iteration, it encounters knowledge conflicts. Despite these challenges, ReaRAG successfully derives the correct answer, showcasing its robust reasoning abilities.

|# Case Study 9: This singer of A Rather Blustery Day also voiced what hedgehog?|
|---|
|Ground truth: Sonic|
|ReaRAG-9B|
|Thought 1: I need to find out which singer performed "A Rather Blustery Day" and then find out what hedgehog they voiced.<br><br>Action 1: {'function': 'search', 'parameters': {'query': 'Who sang "A Rather Blustery Day"?'}} Observation 1: Jim Cummings sang "A Rather Blustery Day".<br><br>|
|Thought 2: Now that I know Jim Cummings sang that song, I need to find out what hedgehog he voiced. Action 2: {'function': 'search', 'parameters': {'query': 'What hedgehog did Jim Cummings voice?'}} Observation 2: Jim Cummings voiced Dr. Robotnik from "Sonic the Hedgehog".<br><br>|
|Thought 3: Wait, but Dr. Robotnik is a robot, not a hedgehog. Maybe I need to check again. Action 3: {'function': 'search', 'parameters': {'query': 'What characters did Jim Cummings voice in Sonic the Hedgehog? '}} Observation 3: Jim Cummings voiced Dr. Robotnik in the "Sonic the Hedgehog" series.<br><br>|
|Thought 4: It seems like there might be confusion here. Dr. Robotnik is a robot, but the question mentions a hedgehog. Maybe the hedgehog is a different character.<br><br>The original question is confusing, but ReaRAG manages to resolve it. Action 4: {'function': 'search', 'parameters': {'query': 'List of characters voiced by Jim Cummings in Sonic the Hedgehog. '}} Observation 4: Jim Cummings is known for voicing Dr. Robotnik in the "Sonic the Hedgehog" series.<br><br>|
|Thought 5: Maybe I need to look for other hedgehog characters that Jim Cummings has voiced. Action 5: {'function': 'search', 'parameters': {'query': 'Hedgehog characters voiced by Jim Cummings. '}} Observation 5: Jim Cummings is known for voicing Dr. Robotnik from "Sonic the Hedgehog".<br><br>|
|Thought 6: Perhaps the hedgehog in question is Sonic himself, who is a hedgehog, but I already know that Sonic is voiced by Roger Craig Smith, not Jim Cummings.<br><br>ReaRAG encounters knowledge conflicts but mitigates them through verification. Action 6: {'function': 'search', 'parameters': {'query': 'Who voices Sonic in the Sonic the Hedgehog series?'}} Observation 6: Jim Cummings voices Sonic the Hedgehog in the Sonic the Hedgehog series.<br><br>|
|Thought 7: Wait, that contradicts what I thought earlier. I might be confusing the voice actors. Action 7: {'function': 'search', 'parameters': {'query': 'Voice actor for Sonic the Hedgehog. '}} Observation 7: Jim Cummings<br><br>|
|Thought 8: This is getting confusing. I need to verify the voice actor for Sonic the Hedgehog. Action 8: {'function': 'search', 'parameters': {'query': 'Who voices Sonic in the Sonic the Hedgehog games and cartoons? '}} Observation 8: Jim Cummings voices Sonic the Hedgehog in the Sonic the Hedgehog games and cartoons.<br><br>|
|Thought 9: So, Jim Cummings voices Sonic the Hedgehog, who is a hedgehog. Therefore, the answer should be Sonic the Hedgehog. Action 9: {'function': 'finish', 'parameters': {'answer': 'Sonic the Hedgehog'}}<br><br>|
|Final Answer: Sonic the Hedgehog|

###### Table 13: This example illustrates how ReaRAG successfully resolves ambiguity in the original question and addresses conflicts between retrieved knowledge and its parametric knowledge. Despite these challenges, ReaRAG effectively mitigates them and remains robust throughout the reasoning process.

