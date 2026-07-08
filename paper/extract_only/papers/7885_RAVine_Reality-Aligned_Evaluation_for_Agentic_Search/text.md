# arXiv:2507.16725v2[cs.CL]31Jul2025

## RAVine: Reality-Aligned Evaluation for Agentic Search

Yilong Xu∗†

ICT, CAS Beijing, China xuyilong23s@ict.ac.cn

Xiang Long∗

ModelBest Inc. Beijing, China longxiang1@modelbest.cn

### Abstract

Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine—a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model’s interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at https://github.com/SwordFaith/RAVine.

### CCS Concepts

• Information systems → Evaluation of retrieval results; • Computing methodologies → Natural language processing.

### Keywords

Retrieval-Augmented Generation, Evaluation, Agentic Search

### 1 Introduction

The emergence of the Retrieval-Augmented Generation [RAG; 10, 17]hasaddressedfactualityissuesin Large Language Models [LLMs; 3, 43] and fundamentally transformed the way people access information. To enable more intelligent RAG systems, the retrievalaugmentation paradigm is undergoing a shift: from static and passive search [2, 30, 40] to agentic search [18, 31], and agent-driven workflow systems like Deep Research [12, 23].

As a model-level capability, agentic search aims to enable adaptive and autonomous retrieval. However, existing evaluation frameworks for LLMs with agentic search are misaligned with this target in several key aspects:

Misalignment between "deep search" and user needs. Some existing benchmarks encourage LLMs with agentic search to uncover answers hidden behind highly complex queries [34], where

∗Equal contribution. †This work was done during the internship at ModelBest. ‡Corresponding author

Zhi Zheng‡

ModelBest Inc. Beijing, China zhengzhi@modelbest.cn

Jinhua Gao

ICT, CAS Beijing, China gaojinhua@ict.ac.cn

the expected output may be a short entity. While this setup partially reflects the model’s ability to perform deep agentic search, it diverges from typical real-world user cases, where queries are often under-specified and lack explicit constraints. In practice, as illustrated in Figure 1, users often expect not only depth but also breadth—seeking to gather and integrate multiple points of information, ultimately leading to a long-form, comprehensive answer.

Misalignment between noisy nugget collection and precise end-to-end evaluation. Tasks that require integrating multipoints information and generating long-form outputs demand finegrained, claim-level evaluations, which frequently include task completeness and faithfulness. Some recent works propose using LLM-as-a-Judge to dynamically generate evaluation rules instead of relying on predefined ground truth [37], which can introduce instability. In grounded evaluations, nuggets1 are often treated as evidence of task completeness [25, 26]. However, present approaches fall short in terms of nugget collection effectiveness and application in fine-grained evaluation. Coelho et al. [5], Qi et al. [27] directly instruct LLM to extract nuggets from long web documents, overlooking the limitations of model capabilities and often resulting in incomplete or inaccurate nuggets. Pradeep et al. [25] alleviate collection difficulty through segment-level batched iteration, but it may result in nugget loss due to coverage issues during iteration and truncation of the maximum number of nuggets. Additionally, these methods assess task completeness and faithfulness independently, since the nuggets are not traceable back to web pages, which can lead to inconsistency and also increased evaluation cost.

Misalignment between end-to-end evaluation and processoriented architecture. While end-to-end evaluation directly reflects overall performance, agentic LLMs with search are inherently process-oriented models: they autonomously iterate, invoke search tools, and read web pages during task execution. However, existing evaluation frameworks focus exclusively on quality assessment of the final report [5, 25], overlooking the intermediate behaviors and efficiency of the process, which can also provide important signals for model differentiation and capability diagnosis. Moreover, unlike classic RAG frameworks with fixed pipelines, the flexible and dynamic iterative processes of agentic LLMs also introduce challenges in runtime efficiency and expenses, particularly for tool invocations and model inference.

To address the misalignments in existing evaluation methods, we propose RAVine—a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine is a comprehensive system, encompassing the web environment, benchmark datasets, and a novel evaluation method, serving as a full-process, reproducible, and goal-aligned evaluation sandbox.

1Nuggets refer to gold information units extracted from gold documents associated with a query; it serves as the claim-level ground truth for evaluation.

[from user logs]

[from BrowseComp]

[Figure 1]

[Figure 2]

🂲 Noisy Nuggets Collection!

###### Agentic LLMs with Search

[Figure 3]

Between 1990 and 1994 inclusive, what teams played in a soccer match with a Brazilian referee had four yellow cards, two for...

[Figure 4]

how bad did the vietnam war devastate the economy in 1968

[Figure 5]

Web Corpus

[Figure 6]

[Figure 7]

[Figure 8]

🂱 Misaligned with User Cases!

[Figure 9]

[Figure 10]

[Figure 11]

Iterative Process

InformationFlow

Nuggetization

🂳 Overlook Process

Evaluation!

[Figure 12]

[Figure 13]

Nuggets

[Figure 14]

[Figure 15]

The Vietnam War cost America $77.4 billion in 1968.

# Title: ... ## Inflation ... ## Tax Surcharge

Report Quality Evaluation

The military cost was 9.5% of GDP in 1968.

[Figure 16]

Ireland v Romania

... ## Summary: ...

[Figure 17]

The War caused a plunge in shopping rates.

~BFS+DFS

~DFS

More Realistic

Output (Report)

- Figure 1: Overview of the three primary misalignments addressed by our work. From left to right: (1) The divergence between narrow benchmark queries and broad, real-world user needs. (2) The challenge of collecting reliable and traceable information “nuggets” for fine-grained evaluation. (3) The tendency of existing frameworks to perform end-to-end evaluation while overlooking the agent’s intermediate process.

### 2 Sandbox for Agentic LLMs with Search

Specifically, we use the TREC 2024 RAG Track [24] queries as our test split, derived from Bing logs and reflecting realistic user behavior in web search scenarios. We employ MS MARCO V2.1 [1, 24], a large-scale dataset of Bing web pages, as our corpus to simulate real-world web conditions and enable fine-grained end-to-end and intermediate evaluation. In our proposed evaluation method, we introduce a nugget-centered evaluation approach to assess report quality, characterized by attributability and flexibility. It enables consistent assessment of task completeness and faithfulness, which avoids noise and reduces evaluation costs. Furthermore, nuggets are collected via dynamic clustering, which allows the number of nuggets to adapt to each query, supporting more realistic evaluation and query-sensitive complexity. Last but not least, our approach includes process-oriented metrics for evaluating the performance of agentic LLMs’ intermediate behavior and usage of search tools. We also add efficiency indicators like latency and cost to provide a more comprehensive assessment of model viability.

Developing a sandbox for agentic LLMs with search is intended to support the reliable, stable, and fine-grained evaluation, and also enable process-oriented assessment. The goal of RAVine is also to design a lightweight and reusable framework that significantly reduces the high costs associated with prior evaluation practices, such as extensive API calls and heavy token consumption in LLM-as-a-Judge settings. We envision this sandbox as a transitional evaluation environment that bridges the gap between model training and real-world deployment on the internet.

### 2.1 Static Web Environment

Ensuring temporally consistent, fine-grained evaluation for longform report generation in dynamic environments is inherently challenging, as real-world user scenarios often lack strong constraints, making it difficult to establish stable ground truths over time. To address this issue, we propose using a static web corpus that approximates a real-world internet environment. Specifically, we adopt MS MARCO V2.1 [24], a deduplicated version of the original MS MARCO V2 corpus [1]. This corpus contains 10,960,555 web documents spanning diverse domains, making it well-suited for use as a simulated environment. Moreover, it includes rich metadata such as URLs, titles, and headings, which facilitates the construction of standardized search tools.

Based on RAVine, we evaluate the performance of a series of models. Our analysis reveals three key findings: (1) current models exhibit limitations in task completeness, faithfulness, and search performance; (2) strong performance during the search process does not necessarily lead to high-quality final answers; and (3) models exhibit a tendency to rely on internal knowledge to generate the final report, which is an unattributable and undesirable behavior that has been overlooked in previous evaluation.

We build indexes over the MS MARCO V2.1 corpus to support the implementation of our search tools. To select an indexing strategy best suited to this corpus, we first constructed a development set from TREC Deep Learning Track [6], including DL21, DL22, and DL23 datasets, which feature a range of multihop, multi-source queries. These datasets include relevance annotations with respect to the MS MARCO V2 corpus, which we map to the corresponding webs in MS MARCO V2.1. We then evaluate a series of embedding models on this development set. Based on their retrieval performance and computational efficiency, we select gte-modernbert-base [41], which offers the best trade-off between accuracy and speed. In addition, we also construct a traditional BM25 index [29] to facilitate robustness testing.

In summary, our core contributions are as follows:

- • We propose RAVine, a novel evaluation framework for agentic search, designed to address the misalignment issues in existing evaluation methodologies.
- • We introduce a comprehensive evaluation system that encompasses multiple dimensions, including fine-grained endto-end assessment, intermediate process evaluation, and efficiency analysis.
- • We conduct extensive experiments across a range of LLMs, yielding valuable insights that offer meaningful directions for future research on agentic search.

Dev set Test set

Agentic LLMs with Search

Split name DL21 DL22 DL23 RAG24.test Data volume 53 76 82 86

Corpus

Content

Web List

Table 1: Statistics of the development and test sets. Note that all data mentioned here are those with available qrels, excluding instances from the original dataset that lack qrels.

[Figure 18]

Fetch Search

[Figure 19]

split. These queries are sourced from Bing Search logs and have been manually curated by annotators, often requiring multi-faceted and multi-source information to answer. Moreover, we can obtain relevance annotations for MS MARCO V2.1 corpus of these queries. Specifically, it labels relevant segments—passages consisting of 5 to 10 sentences within their web documents, which allows us to support segment-level nugget extraction. It can also be mapped to document-level relevant annotations to produce gold web pages for process and faithfulness evaluation. Due to identified quality issues in the qrels of two instances discovered during the nugget extraction phase, we exclude these two queries from the final test set. As a result, the final test set consists of 84 queries. For more details, please refer to the Appendix A.1.

Query

URL

Question Answer (Report)

- Figure 2: The architecture of the agentic LLM with search. The core model (M) iteratively interacts with a web Corpus by invoking two primary tools: search, which retrieves a list of web pages for a given query, and fetch, which retrieves the content for a given URL. This cyclical process continues until the agent has gathered sufficient information to produce the final answer.

### 3 Nugget-Centered Report Quality Evaluation

To address the issue of end-to-end evaluation noise, we propose an attributable nugget collection method (§3.1) and a block-level evaluation approach (§3.2) in this section.

- 2.2 Modeling In RAVine, we define agentic LLMs with search as comprising a

core model M and a set of callable search tools Ttool. Given a user query 𝑞, the model M autonomously performs iterative thinking,

interacting with tools in Ttool over multiple rounds, and ultimately outputs a final answer𝑦. The end-to-end process is formally defined as follows:

𝑦 = M (𝑞; Ttool) . (1) The iterative process at step 𝑡 is defined as follows:

(𝜏𝑡,𝑝𝑡) = M (𝑞,𝐻𝑡−1) ,𝜏𝑡 ∈ Ttool, (2)

𝑜𝑡 = 𝜏𝑡 (𝑝𝑡) , (3) 𝐻𝑡 = 𝐻𝑡−1 ∪ {(𝜏𝑡,𝑝𝑡,𝑜𝑡)}, (4)

where 𝑝𝑡 denotes the parameters for tool 𝜏𝑡 generated by M, and 𝑜𝑡 represents the result returned by the tool call. 𝐻𝑡 indicates the historical context up to round 𝑡.

We define two types of tools: search and fetch, which together approximate typical user behavior when browsing the web. Conceptually, they can be viewed as a set of basis operations that span the space of web search behaviors. Specifically, the search tool takes a query as input and returns the top-𝑘 most relevant webs from the corpus, and the fetch tool takes a URL as input and returns the corresponding web document content, as shown in Figure 2. We adopt the standard tool calling format defined by OpenAI in our implementation, which facilitates easier extension and generalization to other scenarios. Details are provided in Appendix A.5.

- 2.3 Test set aligned with real user behavior

### 3.1 Attributable Nuggets Collection

Previous approaches for evaluating task completeness have often relied on document-level nugget extraction. However, a document is a relatively coarse-grained unit that is often lengthy, making it challenging for LLM-Judge to handle effectively. In contrast, a segment is a relatively fine-grained unit that is more aligned with claim statements. Pradeep et al. [25] adopt a segment-level nuggets mining approach. Nevertheless, their iterative strategy, which updates a fixed-size nugget list, might become unstable, causing previously extracted important nuggets to be overwritten, resulting in information loss. Additionally, size restriction of nugget lists can also lead to inaccurate evaluations, as many queries yield more nuggets than the predefined limit. For faithfulness evaluation, we expect that the contents of the report matching the nuggets should cite the original web pages from which those nuggets were extracted. However, existing nugget extraction methods are non-attributable, and must evaluate task completeness and faithfulness separately, which may result in inconsistency and distortion in evaluation.

To address these issues, we propose an attributable and more flexible nugget collection method, as illustrated in Figure 3. To mitigate context length issues that may lead to LLM-Judge performance degradation, we adopt the segment-level extraction strategy, focusing on short segments rather than longer documents that often contain more noise. Both our development and test sets include segment-level qrels, either directly provided or derived via mapping. Next, when constructing batches, we group segments originating from the same document and input them into the LLM-Judge for

To enable evaluation that better reflects real-world user scenarios, we use the queries from the TREC 2024 RAG Track [24] as our test

Attributable Nuggets Collection

Nugget-Centered Report Quality Evaluation

[Figure 20]

Nuggets (vital/okay)

Block-wise Calculation

...

- • Fully Supported
- • Partially Supported
- • Not Supported

[Figure 21]

[Figure 22]

[Figure 23]

q

[Figure 24]

Nuggets

[Figure 25]

Task Completeness

Assignment

[Figure 26]

Report

Claims

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

split

Block i

Gold Citations

...

projection

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Nuggets

...

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Block 3

[Figure 44]

Faithfulness

Block 2 Block 1

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Citations

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

seg seg seg

seg seg seg

seg seg seg

[Figure 60]

[Figure 61]

[Figure 62]

Batch3

Batch1

Batch2

Process-Oriented Evaluation

Tool Performance

Efficiency

[Figure 63]

[Figure 64]

[Figure 65]

Recall/Precision

Generation Latency

...

doc 1 doc 2 doc 3

Search

[Figure 66]

[Figure 67]

[Figure 68]

Tool Call Gain

Tools

Fetch

q segseg segseg segseg

Tool Call Cost

Tool Call Error

...

seg seg seg

- Figure 3: Overview of the evaluation framework of RAVine. Left (§3.1): Attributable nugget collection via batched extraction at segment-level and semantically merging for refined, query-specific nuggets. Top-Right (§3.2): Block-level evaluation measuring task completeness and faithfulness of the final report. Bottom-Right (§4): Process evaluation of tool performance and efficiency.

nugget extraction. Based on our investigation, the number of relevant segments per document is typically limited, which helps keep the context length within manageable bounds. Moreover, since MS MARCO V2.1 employs a sliding window strategy for segmentation which results in overlapping content between adjacent segments, grouping segments from the same document into a batch enables the LLM to perform effective deduplication. Most importantly, because each batch corresponds to a specific web page, the nuggets generated from it can be reliably attributed to their source, providing a robust foundation for attributable nuggetization.

with an excessively large number of nuggets in order to mitigate evaluation costs. This strategy is more flexible and generalizable, allowing the evaluation data to retain a diverse range of information densities. Further details are provided in Appendix A.2.

Following Pradeep et al. [25], we instruct the LLM-Judge to perform batched nugget scoring to support task completeness evaluation. Each nugget is labeled as either vital, indicating that it is highly important and must be included in the report, or okay, meaning it contains good-to-have information for the query but is not essential.

Next, we project all initially extracted nuggets into a semantic space using an embedding model and perform dynamic clustering. Specifically, we apply HDBSCAN [22], a density-based clustering algorithm, to group semantically similar nuggets into clusters and identify outliers—i.e., nuggets that are relatively dissimilar to others. For each cluster, we prompt the LLM to merge similar nuggets within the cluster and attribute the merged nugget to all original nuggets in that cluster. This design allows a single nugget to be associated with multiple source webs, reflecting the information redundancy and sharing commonly observed in real-world web environments. For each query, we treat the merged nuggets from all clusters, along with the identified outliers, as the final nugget list. This approach offers two key benefits: (1) semantic merging helps dynamically control the total number of nuggets and reduces the difficulty of LLM-based merging, and (2) filtering out outliers further reduces the computational overhead for LLM-based merging.

### 3.2 Block-level Assignment

With the availability of attributable nuggets, we can perform a joint evaluation of task completeness and faithfulness to mitigate inconsistencies and distortions. The first step is to select an appropriate granularity for segmenting the long-form report, which serves as the basic evaluation unit. Prior evaluation methods typically adopt sentence-level granularity [9, 21], and some of them penalize sentences without citations [5, 25]. However, not all sentences in a report constitute factual statements, as some may reflect common sense or general knowledge that does not require citation. Moreover, a complete factual statement may span multiple sentences. Existing approaches often rely on LLMs to directly extract factual statements [25, 27], but this can introduce noise and reduce reliability.

To enable a more flexible evaluation, we adopt block-level granularity. A block refers to a text segment obtained by splitting the report at citation boundaries. Each block is thus associated with the citation(s) that appear at its end. Compared to individual sentences, blocks are more likely to contain semantically complete

At the dataset level, we analyze the distribution of nugget counts across different queries in the test set. Instead of applying a fixed heuristic cutoff to all queries, we selectively truncate only those

factual statements. Additionally, we instruct the model to generate citations only after completing a factual statement in the final report, which is consistent with the behavior of most agentic search systems, such as Deep Research [12]. As illustrated in Figure 3, we evaluate each block’s support for the nuggets to calculate task completeness. Following prior work, we categorize the support level into three classes: fully supported (1 point), partially supported (0.5 point), and not supported (0 point), and prompt the LLM-Judge to assign these labels. In the calculation, we assign lower weights to okay nuggets to reflect their relatively lower importance. The task completeness score is computed as follows:

𝑠𝑣 + 0.5 ∗ 𝑠𝑜

ScoreComp =

, (5)

|𝑁vital| + 0.5 ∗ 𝑁okay

where 𝑠𝑣 and 𝑠𝑜 represent the assigned score of each vital or okay nugget, respectively.

After obtaining the nugget judgments for each block, we trace the matched nuggets back to their corresponding source webs. These webs then serve as the ground truth citations that should have been cited in the corresponding block. Based on this, we compute the citation recall and citation precision for each block. The formulas are as follows:

𝐶Pred ∩𝐶Nugget 𝐶Nugget

𝐶Pred ∩𝐶Nugget 𝐶Pred

Recall(𝑖)

, Precision(𝑖)

cite =

cite =

,

(6) where 𝐶Pred and 𝐶Nugget denote the generated citations and the ground truth citations of block 𝑖, respectively. Additionally, we standardize the citation format by instructing the model to generate citations in Markdown style, i.e., in the form of [title](url), and multiple citations are allowed to appear in the same position.

With the proposed method, computing block-level task completeness and faithfulness simultaneously addresses the inconsistency and distortion issues present in prior non-attributable nugget-based evaluation approaches. And the final report-level score is obtained by averaging the performance across all blocks. Moreover, since our method requires only one round of nugget assessment over the entire report and does not rely on the LLM-Judge for any additional assessments, it significantly reduces the evaluation cost.

### 4 Process-Oriented Evaluation

The architecture of agentic LLMs with search is inherently procedural, as the model engages in an iterative process of autonomous reasoning and tool invocation. Given this characteristic, evaluation should account for two key aspects: (1) performance of the model’s interaction with external tools, and (2) efficiency and computational cost, as the unfixed and potentially prolonged iterative process can incur substantial overhead. Thus, even if the model achieves satisfactory end-to-end performance, poor tool usage, low efficiency, and excessive cost may cast concerns on its practical utility.

### 4.1 Tool Call Performance

First, we evaluate the correctness of generated tool calls. For tool selection, we examine whether the model makes inappropriate choices during the iterative process, such as invoking the fetch tool instead of search in the first step or calling undefined tools.

For parameter correctness, we verify whether the model-generated parameters conform to the tool’s definition and whether any undefined parameters are included. In particular, the fetch tool introduces a specific correctness requirement: whether the provided URL is valid. These metrics collectively assess the fundamental capabilities of agentic LLMs with search across multiple dimensions.

Subsequently, we evaluate the effectiveness of tool calls, assessing how much useful information each invocation contributes to the task. By leveraging the qrels, which we denote as 𝑅𝑒𝑙 (𝑞) for query 𝑞, we can identify the set of relevant web pages. For the search tool, we suppose there are𝑇𝑠 total calls. The returned top-𝑘 result of the 𝑡-th call is denoted as 𝐷𝑡 = {𝑑𝑡1,𝑑𝑡2, . . .,𝑑𝑡𝑘}, where each element represents a retrieved web page. We introduce the notion of marginal information utility, which refers to the incremental contribution of each new search invocation to the overall coverage of relevant information. We first define the Cumulative Relevant Information Coverage 𝐶 at each search round as follows:

𝑡 𝑖=1 (𝐷𝑖 ∩ 𝑅𝑒𝑙 (𝑞))

. (7)

𝐶 (𝑡) =

|𝑅𝑒𝑙 (𝑞)|

Next, the search gain, i.e., the marginal information gain brought by each invocation of the search tool, can be calculated as follows:

𝛿search (𝑡) = 𝐶 (𝑡) −𝐶 (𝑡 − 1) = (𝐷𝑡 ∩ 𝑅𝑒𝑙 (𝑞)) \ 𝐷<seen𝑡

, (8)

|𝑅𝑒𝑙 (𝑞)|

where 𝐷<seen𝑡 = 𝑖 𝑡=−11𝐷𝑖, which denotes the set of web pages retrieved in all previous rounds before step 𝑡. To evaluate the overall

performance of the search behavior throughout a run, we aggregate the per-round search gains across all rounds, as defined below:

##### ∑︁𝑇

𝛿¯search =

𝜔 (𝑡) · 𝛿search (𝑡) , (9)

𝑡=1

where 𝜔 (𝑡) denotes the weight assigned to the 𝑡-th round. In our setup, we define 𝜔(𝑡) = 𝑇1 , treating all tool invocation steps equally with uniform weighting.

In addition to search gain, which captures the information gain at each step of the iterative process, we also consider the model’s overall search performance. Based on the definitions above, we compute the search recall and precision of the entire run with respect to the relevant information, as follows:

𝑇 𝑖=1 (𝐷𝑖 ∩ 𝑅𝑒𝑙 (𝑞))

Recallsearch =

= 𝐶 (𝑇) , (10)

|𝑅𝑒𝑙 (𝑞)|

𝑇 𝑖=1 |𝐷𝑖 ∩ 𝑅𝑒𝑙 (𝑞)|

Precisionsearch =

, (11)

𝑇 𝑖=1 |𝐷𝑖|

where Recallsearch reflects the accuracy of the model’s query trajectory in covering relevant information, and Precisionsearch reflects its efficiency, measuring how often retrieved pages are relevant. For the fetch tool, due to the model’s limited context length and the fact that useful information may already exist in the titles or headings returned by the search results, the model may choose not to fetch all relevant pages. As a result, fetch recall may not be a meaningful metric at this stage. Therefore, we define only the fetch precision metric, which evaluates whether each fetched page is relevant: a score of 1 is assigned if the fetched page is relevant, and 0 otherwise.

- Table 2: Evaluation results on RAVine, with a maximum context length of 32k and the index built by gte-modernbert-base. "Rate" denotes the Task Completion Rate. "Comp." refers to the score of Task Completeness. "Rec." and "Prec." represent Recall and Precision, respectively. "URL Err." denotes the URL Error. Latency is measured in seconds, and cost is measured in dollars. Symbols (↑) and (↓) indicate that higher or lower values are preferred, respectively. Models not marked with (Thinking) either run without thinking or lack support for the thinking mode. Bold values indicate the best performance for each corresponding metric in the column.

Report Quality Efficiency Search Fetch

Rate (↑) Comp. (↑) Rec. (↑) Prec. (↑) Latency (↓) Cost (↓) Turns Prec. (↑) Rec. (↑) Gain (↑) URL Err. (↓) Prec. (↑)

Qwen2.5-7B-Instruct 19.0 6.8 1.9 1.8 7.1 0.01 3.1 18.7 5.7 4.7 8.8 19.8 Qwen2.5-32B-Instruct 71.4 23.0 14.9 16.5 40.3 0.03 4.0 21.1 6.5 4.4 1.4 28.7 Qwen3-8B (Thinking) 86.9 37.8 10.4 12.1 13.9 0.03 6.6 19.7 6.6 5.1 8.9 27.3 Qwen3-8B 28.6 12.4 4.8 6.1 11.2 0.06 9.3 19.3 5.9 5.0 2.4 23.8 Qwen3-32B (Thinking) 98.8 43.5 11.7 15.1 19.6 0.02 2.8 19.2 5.0 4.0 8.9 22.2 Qwen3-32B 85.7 38.0 12.8 12.6 14.6 0.08 8.5 19.1 6.3 5.0 8.1 20.2 Qwen3-30B-A3B (Thinking) 81.0 35.6 10.6 14.2 33.0 0.10 6.6 19.7 6.2 3.6 10.3 29.3 Qwen3-30B-A3B 77.4 30.9 11.3 14.2 15.7 0.07 7.3 16.8 6.2 3.4 0.6 30.4 LLaMA-3.1-8B-Instruct 96.4 24.0 3.1 3.1 7.3 0.02 2.7 12.1 8.8 6.6 36.8 15.8

- Table 3: Evaluation results on RAVine, with a maximum context length of 128k and the index built by gte-modernbert-base. Other descriptions follow Table 2.

Report Quality Efficiency Search Fetch

Rate (↑) Comp. (↑) Rec. (↑) Prec. (↑) Latency (↓) Cost (↓) Turns Prec. (↑) Rec. (↑) Gain (↑) URL Err. (↓) Prec. (↑)

Qwen2.5-7B-Instruct 1.2 0.3 0.0 0.0 4.5 0.01 1.6 7.4 2.6 2.3 0.0 33.3 Qwen2.5-32B-Instruct 61.9 24.5 9.7 11.8 17.9 0.03 3.8 19.7 6.0 4.3 3.6 28.3 Qwen3-8B (Thinking) 91.7 41.9 8.3 10.3 65.8 0.40 23.3 12.9 6.3 2.2 5.6 25.1 Qwen3-8B 26.2 10.1 2.6 3.8 139.5 2.52 113.3 8.0 6.3 2.3 4.1 23.8 Qwen3-32B (Thinking) 100.0 45.2 8.4 9.6 23.0 0.02 2.7 18.6 5.2 4.1 0.0 8.6 Qwen3-32B 82.1 35.0 13.2 11.9 22.7 0.42 14.8 15.0 7.0 3.2 6.5 20.3 Qwen3-30B-A3B (Thinking) 81.0 36.8 10.9 12.0 46.9 0.43 12.1 19.0 6.3 4.7 3.8 12.1 Qwen3-30B-A3B 46.4 16.9 6.1 6.7 54.2 0.64 16.7 18.5 6.4 4.8 1.9 23.1 LLaMA-3.1-8B-Instruct 98.8 25.8 2.8 3.3 4.4 0.02 2.7 12.3 8.0 6.3 58.2 10.9

### 4.2 Efficiency and Cost

We primarily consider two metrics: latency and monetary cost. Latency is used to reflect the efficiency for iteration, tool invocation, and answer integration of the model. For monetary cost, we specify the per-token price for prompts and for completions, as well as a fixed cost to each type of tool invocation.

### 5 Experiments

In this section, we evaluate the search capabilities of several agentic LLMs with search in our proposed RAVine framework.

### 5.1 Experimental Setup

In our proposed RAVine framework, we build both dense and lexical indexes for the web environment on which the search and fetch tools rely. Specifically, we use the gte-modernbert-base

- [41] model to build the dense index, implemented with FAISS [7], and adopt BM25 [29] for the lexical index, implemented using Pyserini [20]. In the nugget-centered report quality evaluation, we use the gemini-2.5-flash (without thinking) [11] as the LLM in both nugget extraction and report scoring. All the prompt templates in the process can be found in Appendix A.3.

We select several state-of-the-art agentic LLMs with tool-use capabilities for evaluation, including Qwen2.5-7B-Instruct [28], Qwen2.5-32B-Instruct[28], Qwen3-8B[38], Qwen3-32B[38], Qwen330B-A3B [38] and LLaMA-3.1-8B-Instruct [13]. For models that support a thinking switch, such as the Qwen3 series, we evaluate their performance under both thinking and non-thinking modes. We divide the evaluation of all models into two groups based on the maximum context length: 32k and 128k. For each group, we conduct separate evaluations using the two different indexes. All experiments are conducted on a machine with 8 × H100 GPUs.

### 5.2 Metrics

In addition to the defined metrics in §3.2, the specific definitions of other metrics are as follows:

- • Task Completion Rate: The proportion of runs in which the model successfully follows instructions, completes the iterative process, and generates a final report.
- • Latency: Measured as the total wall time from the beginning to the end of each run.
- • Cost: Estimated using predefined unit prices, following the pricing scheme of Together.ai2.

2https://www.together.ai/pricing

Qwen3-8B, 32k

Qwen3-8B, 128k

Qwen3-32B, 32k

Qwen3-32B, 128k

1.0

Internal (r=0.05, p=0.753)

Internal (r=0.06, p=0.685)

Internal (r=-0.04, p=0.777)

Internal (r=0.06, p=0.667)

1.0

1.0

Search (r=0.32, p=0.09)

Search (r=0.30, p=0.13)

Search (r=0.60, p=0.00014)

Search (r=0.57, p=0.00362)

0.8

Overall (r=0.22, p=0.064)

Overall (r=0.19, p=0.0993)

Overall (r=0.24, p=0.0299)

Overall (r=0.25, p=0.0219)

0.8

0.8

0.8

SearchPrecision

SearchPrecision

SearchPrecision

SearchPrecision

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0.2

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

0.0 0.2 0.4 0.6 0.8

Task Completeness

Task Completeness

Task Completeness

Task Completeness

- Figure 4: Correlation analysis between Task Completeness and Search Precision. Blue and orange points represent runs based on internal knowledge and search, respectively, and are used to fit the blue and orange regression lines. The red line represents the regression fit over all data points. In the top-left legend, r denotes the Pearson correlation coefficient and p indicates statistical significance. Overall trend shows weak positive correlation, while search-based runs show stronger positive correlation.

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

0

10

20

30

40

50

Percentage(%)

32k

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

0

10

20

30

40

50

Percentage(%)

128k

- Figure 5: Statistics of the proportion of task completeness

Limited task completeness in model performance. Some models, such as Qwen2.5-7B-Instruct and Qwen3-8B without thinking, exhibit relatively low task completion rates, highlighting limitations in their ability to follow instructions. The completion rate reflects a model’s capacity to correctly interpret and execute a task through appropriate tool usage, structured reasoning, and coherent answer synthesis. Failure to follow critical instructions at the beginning during iterative running often leads to incomplete task execution. Moreover, excessive or redundant search queries can cause the context length to exceed the model’s limit, resulting in task failure. This further exposes deficiencies in the model’s search planning and precision. In addition, most models fail to autonomously retrieve a sufficient amount of relevant information. For instance, Qwen3-8B without thinking recalls only 22.3% of the nuggets on average, falling short of the requirements for generating comprehensive, in-depth long-form responses. These findings suggest that the agentic search capabilities of current models still require significant improvement.

attributed to internal knowledge (Compin) across models. The experiment is conducted using the dense index. “Q” denotes Qwen, and “L” denotes LLaMA. The numbers indicate the version and parameter scale. “T” refers to the thinking mode, while “N” refers to the mode without thinking.

Limited faithfulness in model performance. Under the classic, static RAG paradigm, where the context containing citation sources is explicitly presented, LLMs have demonstrated strong citation capabilities [9, 35]. However, our experimental results indicate that LLMs with agentic search still exhibit significant limitations in citation generation. For instance, Qwen3-32B can only achieve a maximum citation recall of 13.2% and precision of 11.9%, respectively. We attribute this to the increased difficulty of context management introduced by multi-turn search, which complicates the tracking of information sources and leads to a decline in faithfulness.

- • Turns: The average number of iterations per run, with each LLM’s generation counted as one turn.
- • URL Error: The proportion of fetch tool calls that result in URL errors.

Please refer to Appendix A.4 for more details of metrics.

### 5.3 Results

The evaluation results on the dense index under the 32k and 128k context length settings is presented in Tables 2 and 3, respectively. Results on the BM25 index can be found in Appendix B.1. We observe that models show inconsistent performance across two indexes. For details of robustness experiment, please refer to Appendix B.2. Moreover, we find that all models exhibit a good understanding and adherence to the basic usage of tools, with few noticeable errors in parameter formatting or tool selection. As such, we do not report these statistics in the tables.

Limited query exploration capability. The metrics search recall and search gain are used to evaluate the model’s use of the search tool from two perspectives: overall coverage and marginal utility. Specifically, a search gain of 4.7 indicates that, on average, each invocation of the search tool retrieves 4.7% of new relevant information (i.e., documents). Search recall measures the proportion of all relevant information retrieved throughout the entire iterative process. The evaluated models perform poorly on both metrics, suggesting that they have limited query expansion and exploration

Several key insights and analyses are presented below:

capabilities and may linger in a certain information domain during running. An effective search behavior should achieve both high recall and high marginal gain, demonstrating both breadth and depth in information seeking.

"Thinking" enhances agentic search. Under the thinking mode, models generally achieve higher task completion rates and task completeness scores, as the thinking mode enables the model to develop a deeper understanding of the user query and the required information, leading to more deliberate and effective tool use. Although thinking mode introduces additional latency and costs in token generation, the mode without thinking often results in more aggressive and inefficient tool calls, such as Qwen3-8B without thinking, which significantly increases the overall cost.

Attribution is impacted by reliance on internal knowledge. When performing factual tasks, LLMs may rely not only on retrieved external knowledge but also on their internal knowledge to generate responses. To investigate this behavior, we calculate the proportion of task completeness scores that relies on internal knowledge, and denote this metric as Compin. Specifically, if a model fails to retrieve any web pages that hit the qrels during the iterative process, we consider the final score of this run to be based on internal knowledge, and this score contributes to Compin. As shown in Figure 5, a significant portion of task performance across models can be traceable to internal knowledge. For some models, more than half of the nuggets are hit by internal knowledge. While such internal knowledge may be accurate for some non-long-tail facts, it cannot be attributable through citations. Such behavior can impair the faithfulness and verifiability, which is undesirable in the context of search-augmented systems and has been overlooked by previous evaluation frameworks.

The illusion of progress: Intermediate performance does not always translate to final outcomes. As shown in Figure 4, we observe no significant positive correlation between the final task completeness scores and search precision scores. To investigate this phenomenon, we employ the Compin metric and categorize each run into those relying on internal knowledge versus those relying on retrieved knowledge. Linear regression analysis reveals that runs depending on retrieved knowledge exhibit a more pronounced positive correlation between task completeness and search precision. This suggests that excessive reliance on internal knowledge is one of the factors contributing to the discrepancy between intermediate and final performance. Moreover, if the model is capable of incorporating effective search performance into the final report, our framework can encourage this good behavior through higher scores.

### 6 Related Work

Retrieval-Augmented Generation. Retrieval-Augmented Generation [RAG; 10, 17] enhances Large Language Models [LLMs; 43] by incorporating external knowledge retrieval, thereby mitigating hallucinations that arise when relying solely on internal parameters

- [42]. Beyond producing accurate and up-to-date answers, RAG also improves the faithfulness and verifiability of generated content by attributing information to source documents via citations [9, 21, 35]. However, traditional RAG paradigms [30, 36, 40] rely on fixed retrieval pipelines that are overly dependent on the initial query and

lack interaction with the retrieved content, resulting in suboptimal performance.

Agentic Search and Deep Research. To enable a more autonomous and adaptive RAG paradigm, agentic search has emerged as a novel approach to augment generation [18, 31]. In contrast to traditional RAG methods that heavily rely on retrieval performance, agentic RAG leverages LLMs to actively guide the search process and reflect on subsequent actions [15, 19]. To better support customized information needs, Deep Research systems adopt a product-level design that implements a carefully structured agent workflow for more comprehensive and targeted search [12, 23, 44]. As a core modellevel capability, agentic search plays a crucial role in determining both the performance of downstream RAG tasks and the upper bound of search agent capabilities. However, existing evaluation frameworks are often misaligned with the goals of agentic search and real-world user search tasks.

Evaluation. Traditional RAG datasets primarily focus on multihop QA tasks involving the retrieval of factual entities [4, 14, 33, 39]. These datasets are typically constructed from some Wikipedia dumps, which suffer from outdated content and exhibit a significant distributional bias compared to real-world search corpora. To support more challenging search evaluations, Wei et al. [34] employ highly complex queries to assess a model’s deep retrieval capabilities. However, these benchmarks often diverge from real-world user search scenarios, where queries tend to be less constrained and users typically expect long-form, comprehensive answers [8, 16]. Moreover, current evaluation frameworks may introduce noise when extracting fine-grained ground truth units—i.e., nuggets, due to limitations in LLM capabilities and information loss, which in turn leads to distorted evaluation outcomes [5, 25]. In addition, many existing frameworks focus solely on end-to-end performance, overlooking critical aspects of agentic search such as intermediate thinking and tool invocation, which are essential to understanding and improving model behavior.

To address the limitations of prior related work, we propose RAVine, a Reality-Aligned eValuation framework for agentic LLMs with search capabilities.

### 7 Conclusion

In this work, we propose RAVine, a novel and comprehensive evaluation framework for agentic search. RAVine is designed to address the misalignment between existing evaluation methods and agentic search’s goal capabilities and real-world user needs, offering a complete suite including the web environment, benchmark data, and the evaluation methodology. RAVine supports both end-to-end evaluation and process-oriented assessment. Notably, we propose an attributable nuggets collection method and a block-level evaluation strategy, which together enable more accurate and cost-efficient fine-grained assessments. Our process-level evaluation metrics are designed to assess a model’s use of search tools, thereby capturing the essential competencies of agentic search. Through our experimental results and analysis, we discover several limitations in current models, particularly in information gathering and faithfulness. Moreover, our process-level evaluation reveals a misalignment

between intermediate behavior and final performance, and uncovers an undesirable behaviors where the model relies on internal knowledge instead of search. We hope that RAVine, along with the insights presented in this work, can contribute to advancing the research and development of agentic search systems.

### References

- [1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs.CL] https://arxiv.org/abs/1611.09268
- [2] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from trillions of tokens. arXiv:2112.04426 [cs.CL] https://arxiv.org/abs/2112.04426
- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. arXiv:2005.14165 [cs.CL] https://arxiv.org/abs/2005.14165
- [4] Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. 2020. HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data. In Findings of the Association for Computational Linguistics: EMNLP 2020, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 1026–1036. doi:10.18653/v1/2020.findingsemnlp.91
- [5] João Coelho, Jingjie Ning, Jingyuan He, Kangrui Mao, Abhijay Paladugu, Pranav Setlur, Jiahe Jin, Jamie Callan, João Magalhães, Bruno Martins, and Chenyan Xiong. 2025. DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research. arXiv:2505.19253 [cs.IR] https://arxiv.org/abs/ 2505.19253
- [6] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin.

2021. Overview of the TREC 2021 Deep Learning Track. In TREC. https://trec. nist.gov/pubs/trec30/papers/Overview-DL.pdf

- [7] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The Faiss library. (2024). arXiv:2401.08281 [cs.LG]
- [8] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long Form Question Answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Anna Korhonen, David Traum, and Lluís Màrquez (Eds.). Association for Computational Linguistics, Florence, Italy, 3558–3567. doi:10.18653/v1/P19-1346
- [9] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large Language Models to Generate Text with Citations. arXiv:2305.14627 [cs.CL] https://arxiv.org/abs/2305.14627
- [10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL] https://arxiv.org/abs/2312.10997
- [11] Gemini. 2025. gemini-2.5-flash-preview. https://ai.google.dev/gemini-api/docs/ models?hl=zh-cn#gemini-2.5-flash-preview, Accessed on 2025-06-20.
- [12] Google. 2025. Gemini Deep Research. https://gemini.google/overview/deepresearch/. Accessed: 2025-05-08.
- [13] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen,

Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah

Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783

- [14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. arXiv:2011.01060 [cs.CL] https://arxiv.org/abs/2011.01060
- [15] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL] https://arxiv.org/abs/2503.09516
- [16] Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to Progress in Long-form Question Answering. arXiv:2103.06332 [cs.CL] https://arxiv.org/abs/ 2103.06332
- [17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] https://arxiv.org/abs/ 2005.11401
- [18] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large Reasoning Models. arXiv:2501.05366 [cs.AI] https://arxiv.org/abs/2501.05366
- [19] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, JiRong Wen, and Zhicheng Dou. 2025. WebThinker: Empowering Large Reasoning Models with Deep Research Capability. arXiv:2504.21776 [cs.CL] https://arxiv. org/abs/2504.21776
- [20] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations. In Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021). 2356–2362.
- [21] Nelson Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating Verifiability in Generative Search Engines. In Findings of the Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 7001–7025. doi:10.18653/ v1/2023.findings-emnlp.467
- [22] Claudia Malzer and Marcus Baum. 2020. A Hybrid Approach To Hierarchical Density-based Cluster Selection. In 2020 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI). IEEE, 223–228. doi:10.1109/mfi49285.2020.9235263
- [23] OpenAI. 2025. Deep Research System Card. Technical Report. OpenAI. https: //cdn.openai.com/deep-research-system-card.pdf
- [24] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan Nguyen, Daniel Campos, Nick Craswell, and Jimmy Lin. 2024. Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track. arXiv:2406.16828 [cs.IR] https://arxiv.org/abs/2406.16828
- [25] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. 2024. Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework. arXiv:2411.09607 [cs.IR] https://arxiv.org/abs/2411.09607
- [26] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. 2025. The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models. arXiv:2504.15068 [cs.IR] https://arxiv.org/abs/2504.15068
- [27] Zehan Qi, Rongwu Xu, Zhijiang Guo, Cunxiang Wang, Hao Zhang, and Wei Xu. 2024. 𝐿𝑂𝑁𝐺2𝑅𝐴𝐺: Evaluating Long-Context & Long-Form RetrievalAugmented Generation with Key Point Recall. In Findings of the Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 4852–4872. doi:10.18653/v1/2024.findings-emnlp.279
- [28] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
- [29] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (April 2009), 333–389. doi:10.1561/1500000019
- [30] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. REPLUG: Retrieval-Augmented BlackBox Language Models. arXiv:2301.12652 [cs.CL] https://arxiv.org/abs/2301.12652
- [31] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. 2025. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. arXiv:2501.09136 [cs.AI] https://arxiv.org/abs/2501.09136

- [32] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning. arXiv:2503.05592 [cs.AI] https: //arxiv.org/abs/2503.05592
- [33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.

2022. MuSiQue: Multihop Questions via Single-hop Question Composition. arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573

- [34] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese.

2025. BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents. arXiv:2504.12516 [cs.CL] https://arxiv.org/abs/2504.12516

- [35] Yilong Xu, Jinhua Gao, Xiaoming Yu, Baolong Bi, Huawei Shen, and Xueqi Cheng.

2024. Aliice: Evaluating positional fine-grained citation generation. arXiv preprint arXiv:2406.13375 (2024).

- [36] Yilong Xu, Jinhua Gao, Xiaoming Yu, Yuanhai Xue, Baolong Bi, Huawei Shen, and Xueqi Cheng. 2025. Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models. arXiv preprint arXiv:2504.00573 (2025).
- [37] Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. 2025. An Illusion of Progress? Assessing the Current State of Web Agents. arXiv:2504.01382 [cs.AI] https://arxiv.org/abs/2504.01382
- [38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388
- [39] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL] https://arxiv.org/abs/1809.09600
- [40] Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. 2023. AugmentationAdapted Retriever Improves Generalization of Language Models as Generic PlugIn. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 2421–2436. doi:10.18653/v1/2023.acl-long.136
- [41] Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, et al. 2024. mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 1393–1412.
- [42] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models. arXiv:2309.01219 [cs.CL] https://arxiv.org/abs/2309.01219
- [43] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2025. A Survey of Large Language Models. arXiv:2303.18223 [cs.CL] https://arxiv.org/abs/2303.18223
- [44] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments. arXiv:2504.03160 [cs.AI] https://arxiv.org/abs/2504.03160

- A Implementations

- A.1 Test Set

During the nugget extraction phase, we identify two instances where the LLM consistently failed to extract any nuggets from the annotated relevant segments, even after multiple runs. Upon manual inspection, we find that although these segments contain some keyword overlap with the query, they do not provide any truly relevant or useful information. Consequently, we remove these two data from the test set. These instances are listed in Table 4.

Qid Query 2024-96359

how were the non aligned countries affected by boris yeltsin

2024-44059 how has zabriskie point’s climate changed? Table 4: Data removed from the original test set of TREC 2024 RAG Track.

- A.2 Details of Nuggets Extraction and Assignment

In the prior nugget creation method [25], a fixed upper limit is imposed on the number of nuggets extracted by LLMs for each query—for example, retaining only the top 30 nuggets and discarding the rest. We believe this design aims to control evaluation cost, but at the expense of evaluation accuracy.

To address the evaluation distortion caused by such nugget truncation, we conduct a quantitative analysis of the distribution of all nuggets extracted by the LLMs, as shown in Figure 6. The transformation from segments to nuggets and then to vital nuggets represents a progressive filtering process that reduces the amount of information. According to the distribution, 95% of the queries contain no more than 108.84 nuggets—far exceeding the preset limit of 30 in Pradeep et al. [25]. To strike a balance between preserving the majority of informational content and maintaining evaluation efficiency, we set the nugget cap at the 95th percentile (P95) of the vital nuggets distribution, which corresponds to a count of 56.15. We round this to 60 as the final nugget limit. This threshold prevents both excessive evaluation cost from queries with disproportionately many nuggets (which are arguably anomalous) and information loss due to arbitrary truncation. Compared to a fixed limit, this percentile-based, data-driven truncation strategy offers a more principled and adaptive approach.

In the final report quality evaluation, we assign the degree of support for each nugget by calling a state-of-the-art proprietary LLM via its API. Specifically, we employed the then-current version Gemini-2.5-Flash in our implementation, which is gemini-2. 5-flash-preview-05-20. As the latestofficiallyreleasedgemini-2. 5-flash model is identical to the preview-05-20 version, our usage can be regarded as equivalent to the official version. We clarify this point here for completeness.

0.0175

segments

P95=56.15

nuggets

0.0150

vital nuggets

0.0125

0.0100

P95=108.84

Density

0.0075

P95=292.00

0.0050

0.0025

0.0000

100 0 100 200 300 400 Count

Figure 6: Fitted distribution curves of the number of relevant segments, collected nuggets, and vital nuggets per query. The X-axis represents the corresponding count, and the Y-axis indicates the density for each count. The three vertical lines mark the 95th percentiles (P95) of the distributions for the three types of elements, respectively.

### A.3 Prompt Templates

The prompt templates used for nugget extraction, report evaluation, and the execution of agentic LLMs with search are shown below:

#### Prompt Template for Segment-level Batched Nuggets Creation

Given a query and a list of possibly relevant context, generate a list of atomic nuggets of information from the context, so they best provide the information required for the query. Each generated nugget should be a complete and unique statement of a fact from the context (a sentence of about 10 words). A nugget should include a clear subject, verb, and object, and should avoid using pronouns such a "it". A nugget is not simply a salient statement within the context, but also one that helps answer the query. Return only the list of nuggets in a markdown-style python codeblock, and place nuggets in pythonic list format. Ensure the nuggets list has at most {creator_max_nuggets} nuggets (can be less or empty). Return only the most vital nuggets. Search Query: {query} Context:

{context} Search Query: {query} List in the form ["a", "b", ...] and a and b are strings with no mention of ". If no complete statement that is valuable to the query can be found in the context, do not generate low-quality nuggets, and return [] directly. Do not explain and make sure there is no redundant information. Nugget List:

#### Prompt Template for Nuggets Merging

Given a query, please merge its list of atomic nuggets (if necessary) by combining similar nuggets, and return a new list of atomic nuggets. A nugget refers to a semantically complete and unique statement of a fact (a sentence of around 10 words) that helps answer the query. Query: {query} Nuggets List:

{nuggets_list} Your output should be: one nugget per line, and for each nugget, indicate which original nuggets were merged (by listing their indices). Example: nugget_text [1, 2, ...] If there are no similar nuggets in the list, indicating that no merging is needed, simply return: [NO NEED]. Make sure there is no redundant information.

#### Prompt Template for Nuggets Scoring

Based on the query, label each of the {num_nuggets} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget. Search Query: {query} Nugget List: {nugget_list} Only return the list of labels (List[str]). Do not explain. Labels:

Prompt Template for Nuggets Assignment Based on the query and passage, label each of the

{num_nuggets} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget. Search Query: {query} Passage:

{context} Nugget List: {nugget_texts} Only return the list of labels (List[str]). Do not explain. Labels:"

#### Prompt Template for LLMs with Agentic Search

Your task is to generate a report to answer the question provided. During this process, you need to do the following:

- 1. You primarily respond in English.
- 2. You can choose to call known tools and generate the correct parameters according to the tool description.
- 3. You can generate any content that helps you complete the task during the intermediate iteration process according to your needs.
- 4. When you consider the task complete, the last generated content is a long-form report that covers much useful information for the given question.
- 5. In each iteration, you get to choose what to do next (call the search tool or complete the task and generate a final report), and you do not require assistance or response from users.

You need to meet the following requirements for your final long-form report:

- 1. Your long-form report needs to be in markdown format.
- 2. Your long-form report needs to be logically clear, comprehensive in key points, and able to effectively address the given question.
- 3. Your long-form report needs to include citations of the websites retrieved through external search tools.
- 4. In the final output, your report must be enclosed within <report> and </report>, that is, only the content between these tags will be evaluated.

The citations in your final long-form report need to meet the following requirements:

- 1. Citations can only appear at the end of a sentence.
- 2. Citations must follow the Markdown format, including the website’s title and URL, and should be enclosed in brackets. For example: ([title](url)).
- 3. Multiple citations can appear at the same time in one position, separated by semicolons. For example: ([title1](url1); [title2](url2); [title3](url3)).
- 4. A complete statement may contain one or more sentences. Please try to generate citations after the entire statement is presented.
- 5. Do not list the cited websites at the end of the report to avoid unnecessary token usage. Question: {question}

### A.4 Details of Metrics

The implementation details of several metrics defined in RAVine are as follows:

Task Completion Rate. We require the agentic model to complete the report generation within a predefined context length. Typically, the iterative process consists of multiple search tool invocations,

#### Table 5: The token pricing for LLMs in RAVine, based on the rates published at https://www.together.ai/pricing.

Price / 1M Tokens

Model Parameters

Reference

Input Output LLaMA 8B 0.20 0.20 LLaMA 3.3, LLaMA 3.2, LLaMA 3.1, LLaMA 3 models Qwen

8B 0.20 0.20 All other chat, language, code and moderation models 32B 0.08 0.20 All other chat, language, code and moderation models 30B-A3B 0.06 0.20 Mixture-of-experts

#### Table 6: Evaluation results on RAVine, with a maximum context length of 32k and the index built by BM25. Other descriptions follow Table 2.

Report Quality Efficiency Search Fetch

Rate (↑) Comp. (↑) Rec. (↑) Prec. (↑) Latency (↓) Cost (↓) Turns Prec. (↑) Rec. (↑) Gain (↑) URL Err. (↓) Prec. (↑)

Qwen2.5-7B-Instruct 22.6 6.4 4.4 6.3 6.9 0.02 2.9 24.6 12.3 6.9 27.1 29.2 Qwen2.5-32B-Instruct 70.2 26.2 13.0 17.1 16.6 0.04 3.9 24.7 12.9 6.9 5.1 49.5 Qwen3-8B (Thinking) 63.1 25.9 8.3 11.8 10.9 0.04 5.6 21.4 10.8 4.6 6.6 28.0 Qwen3-8B 15.5 7.9 3.5 4.5 6.1 0.14 13.2 18.5 9.3 3.9 1.8 34.6 Qwen3-32B (Thinking) 91.7 40.9 10.2 12.5 20.6 0.04 3.3 20.3 6.9 4.8 0.0 31.6 Qwen3-32B 57.1 26.7 13.2 14.5 10.1 0.09 7.5 23.6 13.8 6.3 9.8 34.0 Qwen3-30B-A3B (Thinking) 76.2 32.6 12.6 18.5 23.3 0.09 6.0 22.5 11.4 5.5 0.0 45.7 Qwen3-30B-A3B 58.3 23.2 12.3 12.7 9.1 0.13 9.1 17.9 11.3 4.5 1.1 48.9 LLaMA-3.1-8B-Instruct 97.6 22.1 3.0 2.5 4.5 0.02 2.6 21.1 14.8 8.6 46.8 25.5

#### Table 7: Evaluation results on RAVine, with a maximum context length of 128k and the index built by BM25. Other descriptions follow Table 2.

###### Report Quality Efficiency Search Fetch

Rate (↑) Comp. (↑) Rec. (↑) Prec. (↑) Latency (↓) Cost (↓) Turns Prec. (↑) Rec. (↑) Gain (↑) URL Err. (↓) Prec. (↑)

Qwen2.5-7B-Instruct 7.1 1.4 0.0 0.0 21.2 0.01 1.8 10.5 3.2 2.1 66.7 0.0 Qwen2.5-32B-Instruct 73.8 28.7 16.8 20.4 16.8 0.04 3.9 26.9 15.1 7.5 5.3 43.0 Qwen3-8B (Thinking) 90.5 40.8 14.4 17.8 26.7 0.23 13.4 15.8 12.8 3.8 4.8 32.5 Qwen3-8B 19.0 6.3 5.2 5.4 39.2 1.87 81.0 7.4 12.2 1.4 5.0 38.4 Qwen3-32B (Thinking) 100.0 47.5 10.5 12.3 22.6 0.03 2.8 17.6 9.5 6.0 8.5 19.1 Qwen3-32B 76.2 31.1 13.6 17.6 20.6 0.40 13.3 17.3 15.5 4.8 9.2 29.7 Qwen3-30B-A3B (Thinking) 75.0 35.0 10.8 12.8 58.9 0.81 18.5 18.7 13.5 5.3 3.3 29.6 Qwen3-30B-A3B 50.0 18.5 13.1 14.7 35.3 0.37 12.3 20.5 10.0 4.7 6.5 27.1 LLaMA-3.1-8B-Instruct 97.6 23.4 4.6 5.0 9.5 0.02 2.6 23.2 15.6 10.2 28.6 28.6

with the final step dedicated to integrating the retrieved information into a report. Therefore, a run is considered successful only if the report is generated in the model’s final output. Furthermore, in accordance with the instructions, the report must be enclosed within the <report> and </report> tags. If the model fails to follow this format, or if the iteration exceeds the context length limit before generating the report, the run is deemed unsuccessful. We define Task Completion Rate as the proportion of successful runs out of all executed runs.

Citation Recall & Precision. Our citation evaluation is performed at the block level. For each block, we identify the nuggets it hits and extract their associated source web pages to construct the set of gold citations. Since we merge nuggets during the nugget collection stage, a single nugget may correspond to multiple source web pages. To prevent any nugget from having an excessive number of source web pages (e.g., more than three), which could lead the model

to cite an unreasonable number of pages for a single claim (both impractical and of limited utility), we intersect the gold citations with the set of web pages retrieved by the model during its iterative search in the current run. This constraint aligns with the more realistic assumption that a model should only cite from documents it has discovered. In addition, to prevent nuggets from having an excessive number of gold citations, we impose a hard upper limit in the calculation of citation recall: the number of gold citations per nugget is capped at a maximum of three. This approach helps prevent models from generating an excessive number of citations to game the reward from the metric, and also mitigates the risk of information overload.

Previous approaches typically treat all gold citations equally. However, our analysis suggests that different gold citations should be assigned different priorities. For instance, within a single block where multiple nuggets are matched, some web pages may support only one nugget, while others serve as the source for multiple

nuggets simultaneously. In such cases, web pages supporting multiple nuggets should be prioritized for citation. Accordingly, we incorporate this intuition into our implementation of citation recall by introducing weighted gold citations, where each citation is assigned a weight proportional to the number of nuggets it supports. Furthermore, when the number of gold citations exceeds the predefined upper limit, we normalize the total weight by considering only the top three gold citations with the highest weights. A model can achieve full recall only if it cites these top three sources. This strategy ensures that higher-quality citation behavior is rewarded more, while discouraging the model from producing a large number of low-quality citations.

Search Recall. The search recall metric evaluates the model’s ability to retrieve multi-point useful information throughout the entire iterative process. It is computed based on the source web pages associated with the nuggets for each query. To prevent the metric from being diluted in cases where a nugget has a large number of relevant web pages, we also impose the cap of three relevant web pages per nugget during evaluation. For example, if a nugget has more than three relevant sources and the model retrieves any three of them, the search recall for that nugget is considered to be 1. The final search recall for a run is calculated as the average search recall across all nuggets. This approach also effectively discourages the model from repeatedly retrieving documents for the same information point in one run to boost its evaluation score.

Search Gain. For the calculation of search gain, we similarly address the issue of nuggets being associated with a large number of relevant web pages. Specifically, we limit the number of relevant pages per nugget to a maximum of three. As a result, the total set of information gain is further deduplicated and reduced in size. This constraint helps prevent repeated exploration of the same information point and avoids encouraging search behaviors that lead to information overload in the model’s generation.

Cost. We define interface-specific pricing within our sandbox to estimate the overall execution cost, which serves as one of the criteria for model selection. For LLMs’ token pricing, we adopt the rates provided by Together.ai, as shown in Table 5. For the search tool, we follow the pricing of SerpApi3, with each call costing $0.01. The fetch tool is free.

Task Completeness Score on Internal Knowledge. The task completenessscorethatreliesoninternal knowledge, denoted asCompin (as introduced in §5.3), is a heuristic metric designed to reflect the portion of the task completeness score that is purely attributed to the model’s internal knowledge. This metric is computed based on whether the source web pages of the nuggets are retrieved during the search process. If the model’s answer correctly covers certain nuggets whose source documents are not retrieved, it indicates that these nuggets are supported by internal knowledge and are thus non-attributable to the external search. The formal definition of

3https://serpapi.com/pricing

Compin for a nugget in a block is as follows:

1, if 𝐷1seen∼𝑇 ∩ 𝑅𝑒𝑙(𝑞, nugget𝑗) = ∅ and block𝑖 |= nugget𝑗, 0, otherwise.

𝜎𝑖𝑗 =

(12) Then, the Compin score for block𝑖 is defined as:

∑︁𝑁

1 𝑁

Compin(block𝑖) =

𝜎𝑖𝑗, (13)

𝑗=1

where 𝑁 denotes the total number of nuggets. The score for each block is computed accordingly, and the overall Compin (report) score for the entire report (corresponding to the entire run) is obtained by averaging the scores across all blocks.

### A.5 Search Tool Definition

In previous work, search invocations are often implemented by prompting the LLM to generate special tokens such as <search> query</search>[15]or <|begin_of_query|>query<|end_of_query|> [32] to denote search queries. However, this approach can interfere with the model’s native generation capabilities and hinder its adaptability to other tasks or domains. To address this, we adopt the standard tool calling schema provided by OpenAI4, where search calls are encapsulated within a dedicated field in the model’s response. This design preserves the model’s generalization ability. Specifically, the definitions of the search and fetch tools are as follows:

search_tool = { 'type': 'function', 'function': {

'name': 'web_search', 'description': 'Retrieve a list of

documents from the web corpus based on query relevance.',

'parameters': { 'type': 'object', 'properties': {

'query': {

'type': 'string', 'description': ''

}, 'num_results': {

'type': 'number', 'description': 'Number of top results to return.', },

}, 'required': [

'query', 'num_results',

], 'additionalProperties': False ,

} }

}

fetch_tool = { 'type': 'function',

4https://platform.openai.com/docs/guides/function-calling

'function': { 'name': 'web_fetch', 'description': 'Fetch the content of a web

page based on its URL.',

'parameters': { 'type': 'object', 'properties': {

'url': { 'type': 'string', 'description': 'The full URL

of the web page to fetch content from.',

},

}, 'required': [

'url',

], 'additionalProperties': False ,

} }

}

- B Additional Experimental Results

- B.1 Evaluation on BM25 Index

Tables 6 and 7 report the detailed evaluation results of the models using the BM25-based search index under context length limits of 32k and 128k, respectively.

- B.2 Index Robustness Evaluation

To assess the robustness of agentic search, we evaluate models using both dense and BM25 indexes. The differences in search performance across the two indexes for each model are intuitively shown in Figures 7 and 8. We observe that, under the two indexing strategies, several models exhibit substantial differences between their search performance and fetch performance. This phenomenon reflects the models’ varying robustness to different retrieval tools, suggesting that an effective retriever is not necessarily one based on dense semantic indexing. This suggests that the robustness of models to retrieval index variations needs improvement.

Moreover, as shown in Table 8, models consistently achieve higher fetch gain scores under BM25 compared to dense index. We attribute this to the semantic nature of dense indexes, which tend to retrieve semantically similar documents. This indicates that the model has limited ability to distinguish between semantically similar web pages, which reduces its chances of fetching more truly relevant results. Interestingly, when calculating search gain based on original qrels, dense indexes yield higher scores. However, this trend reverse when we used nugget-associated web pages as the basis for relevance. In principle, the BM25 index tends to return more diverse content, whereas dense retrieval typically returns results that are more semantically similar to the query. From the model perspective, this suggests a lack of robustness in query adaptation and traversal across different semantic spaces. From the tool design perspective, it indicates that search tools may need to be customized based on the behavioral characteristics of different models.

32k

128k

| |BM25| |
|---|---|---|
| | | |
| | | |

Dense

Dense

BM25

0.25

SearchPrecision

0.20

0.15

0.10

0.05

0.00

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

- Figure 7: Comparison of search gain scores across models under the two indexes. Model naming follows the rules used in Figure 5.

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

0.0

0.1

0.2

0.3

0.4

0.5

FetchPrecision

32k

Dense

BM25

|Dense| |
|---|---|
|BM25| |

Q2.5-7Q2.5-32Q3-8-TQ3-8-NQ3-32-TQ3-32-NQ3-30-TQ3-30-NL3.1-8

128k

- Figure 8: Comparison of fetch gain scores across models under the two indexes. Model naming follows the rules used in Figure 5.

Table 8: Comparison of average Search Gain and Fetch Gain scores across all evaluated models under different indexes. Nuggets and Qrels respectively indicate whether the retrieved web pages match those identified as relevant in the qrels or those associated with the nuggets. The value with higher score in each comparison is highlighted in bold.

32k 128k dense bm25 dense bm25 Nuggets

Avg. Search Prec. 18.4 21.6 14.6 17.5 Avg. Fetch Prec. 24.2 36.3 20.6 27.6

Avg. Search Prec. 33.7 32.5 27.1 27.0 Avg. Fetch Prec. 44.3 53.0 38.2 40.0

Qrels

### C Case Study

We select two cases from the Qwen3-32B (thinking mode) model running on the dense index with the context length limit of 32k. In the first case, the iterative search performs well, effectively integrating external information to generate a relatively high-quality answer, as illustrated in Table 9. In contrast, the second case fails to retrieve relevant web pages during the search process, resulting in a final answer relying on the model’s internal knowledge, as shown in Table 10.

#### Table 9: Case 1, executed by the Qwen3-32B (Thinking) using a search tool built on a dense index.

Qid: 2024-94706 Model: Qwen3-32B (Thinking) Question: how to views of euthanasia vary from west to east Role: assistant content: <think> Okay, let’s tackle this question about how views on euthanasia differ between the West and the East. First, I need to understand the scope of the question. The user is asking about variations in perspectives, so I should cover cultural, religious, legal, and maybe medical aspects. Starting with the West, I know countries like the Netherlands, Belgium, and Canada have legalized euthanasia under certain conditions. The US has some states where physician-assisted suicide is allowed, like Oregon and Washington. The cultural factors here might include individual rights, secularism, and a focus on personal autonomy. Religious influences in the West are mostly Christian, but there’s a diversity of beliefs, with some denominations opposing euthanasia while others might support it under specific circumstances. In the East, I’m thinking of countries in Asia, the Middle East, and maybe parts of Africa. Here, cultural values like family-centric decision-making and religious beliefs such as Buddhism, Hinduism, and Islam play significant roles. For example, in Japan, there’s a concept of "death with dignity, " but actual legal frameworks for euthanasia are more restrictive. In many Middle Eastern countries, Islamic law (Sharia) prohibits euthanasia, considering it a sin against God’s will. I need to compare these regions. Legal frameworks are a good starting point. The West tends to have more progressive laws, while the East is more conservative, often influenced by religious doctrines. However, there are exceptions. For instance, in some South Asian countries, there might be more nuanced views, but they still lack legal euthanasia. Religious influences are key. In the West, the decline of organized religion in some areas might lead to more secular views, but in others, like the US, religious opposition is strong. In the East, religions like Buddhism emphasize non-harm (ahimsa), which can lead to opposition to euthanasia. Hinduism’s karma and dharma concepts might also influence views, as interfering with natural death could be seen as disrupting karmic balance. Cultural values like collectivism in the East versus individualism in the West affect how euthanasia is perceived. In collectivist societies, family opinions are crucial, whereas in the West, personal autonomy is prioritized. Medical ethics also play a role. In the West, there’s a push for patient rights and autonomy, leading to more support for euthanasia. In the East, the emphasis on family and societal harmony might lead to a more cautious approach, avoiding actions that could disrupt family unity or societal norms. I should also consider recent trends. Some Eastern countries are starting to debate euthanasia more openly, especially with aging populations and increased awareness. For example, South Korea has had discussions about legalizing euthanasia for terminal patients, but it’s still not legal. To gather accurate information, I should use the web_search function to find sources on Western and Eastern perspectives. Then, use web_fetch to get detailed content from relevant articles or legal documents. Need to make sure the sources are credible, like academic journals, reputable news outlets, or official legal documents. Wait, the user wants the report to include citations. So after gathering the info, I need to attribute each point to the sources retrieved. For example, when mentioning the Netherlands’ laws, cite a source from a Dutch government website or a reputable article. Similarly, for religious influences in the East, cite sources from Islamic scholars or Buddhist texts. I should structure the report into sections: Introduction, Western Perspectives, Eastern Perspectives, Comparative Analysis, Recent Trends, and Conclusion. Each section will have key points with citations. The citations need to be placed at the end of relevant sentences, following the specified format. I need to ensure that the report is comprehensive but not too lengthy. Focus on the main differences and underlying reasons. Check for any nuances, like variations within the East (e.g., differences between Japan and India) and within the West (e.g., differences between the US and Europe). Also, verify the legal status in specific countries. For example, in the US, while some states allow physician-assisted suicide, others don’t. Similarly, in the EU, countries like Germany have strict laws against it, while others are more lenient. After collecting all the necessary information and verifying the sources, compile the report, making sure each claim is backed by a citation. Avoid listing all sources at the end; instead, integrate them into the text where appropriate. </think> tool_calls: [

{

"id": "chatcmpl-tool-96da884e41f64f688161ba1bcdb99242", "function": {

"arguments": "{"query": "euthanasia views in Western countries", "num_results": 5}", "name": "web_search"

}, "type": "function"

}

] Role: tool

tool_call_id: chatcmpl-tool-96da884e41f64f688161ba1bcdb99242 name: web_search content: [

{

"title": "Legality of euthanasia - Wikipedia", "headings": "Legality of euthanasia; textbackslash nLegality of euthanasia; textbackslash nContents; textbackslash nTimeline legalization active euthanasia at national level; textbackslash nEuthanasia law by country; textbackslash nArgentina; textbackslash nAustralia; textbackslash nBelgium; textbackslash nThe Belgian parliament legalised euthanasia on 28 May 2002.; textbackslash nIn January 2020, the assizes court in Ghent acquitted three doctors of murder charges for an euthanasia in 2010.; textbackslash nCanada; textbackslash nChile; textbackslash nColombia; textbackslash nDenmark; textbackslash nFinland; textbackslash nFrance; textbackslash nGermany; textbackslash nIndia; textbackslash nMain article: Euthanasia in India; textbackslash nIreland; textbackslash nIsrael; textbackslash nJapan; textbackslash nLatvia; textbackslash nLithuania; textbackslash nLuxembourg; textbackslash nMexico; textbackslash nNetherlands; textbackslash nNew Zealand; textbackslash nNorway; textbackslash nPeru; textbackslash nPoland; textbackslash nPortugal; textbackslash nPhilippines; textbackslash nSouth Korea; textbackslash nSpain; textbackslash nSweden; textbackslash nSwitzerland; textbackslash nTurkey; textbackslash nUnited Kingdom; textbackslash nUnited States; textbackslash nUruguay; textbackslash nNon-governmental organizations; textbackslash nSee also; textbackslash nReferences; textbackslash nWorks cited; textbackslash nExternal links",

"url": "https://en.m.wikipedia.org/wiki/Legality_of_euthanasia"

}, {

"title": "Countries Where Euthanasia is Legal | Factual Facts", "headings": "Countries Where Euthanasia is Legal; textbackslash nCountries Where Euthanasia is Legal", "url": "https://factualfacts.com/countries-euthanasia-legal/"

}, {

"title": "Facts and Statistics on Euthanasia - ReligionFacts", "headings": "Facts and Statistics on Euthanasia; textbackslash nFacts and Statistics on Euthanasia; textback-

slash nReferences; textbackslash nArticle Info", "url": "http://www.religionfacts.com/euthanasia/stats"

}, {

"title": "Religious Views on Euthanasia", "headings": "Religious Views on Euthanasia; textbackslash nReligious Views on Euthanasia; textbackslash nPage Contents; textbackslash nBuddhism; textbackslash nChristianity; textbackslash nCatholicism; textbackslash nProtestantism; textbackslash nChristians in support of euthanasia; textbackslash nHinduism; textbackslash nIslam; textbackslash nJainism; textbackslash nJudaism; textbackslash nShinto; textbackslash nUnitarian Universalism; textbackslash nInfluence of religious views; textbackslash nSee also; textbackslash nAdapted from Wikipedia, the free encyclopedia; textbackslash nShare this; textbackslash nLike this; textbackslash nSearch; textbackslash nQuote of the Day",

"url": "https://slife.org/religious-views-on-euthanasia/"

}, {

"title": "10 Countries Where Euthanasia and Assisted Suicide Are Legal", "headings": "10 Countries Where Euthanasia and Assisted Suicide Are Legal; textbackslash n10 Canada; textbackslash n9 Albania; textbackslash n8 Colombia; textbackslash n7 Japan; textbackslash n6 United States; textbackslash n5 Germany; textbackslash n4 Switzerland; textbackslash n3 Luxembourg; textbackslash n2 Belgium; textbackslash n1 Netherlands",

"url": "https://www.therichest.com/most-influential/10-countries-where-euthanasia-and-assisted-suicide-arelegal/" } ]

#### Role: assistant

content: <think> Okay, so I need to figure out how views on euthanasia differ between the West and the East. Let me start by looking at the search results provided. The first result is from Wikipedia, listing countries where euthanasia is legal. The Netherlands, Belgium, Canada, and others are mentioned. The second result from Factual Facts also lists countries where euthanasia is legal, which aligns with the first one. The third result from ReligionFacts talks about religious views, which is important for understanding cultural differences. The fourth result from slife.org gives detailed religious perspectives on euthanasia, covering Buddhism, Christianity, Hinduism, Islam, etc. The fifth result from The Richest lists countries where euthanasia is legal, including some in the West and even Japan, which is in the East. From the first two results, I can see that Western countries like the Netherlands, Belgium, Canada, and some US states have legalized euthanasia. The Wikipedia entry mentions the Belgian parliament legalizing it in 2002, and the Factual Facts list includes Canada and the US. The fifth result also includes Japan in the list of countries where euthanasia is legal, which is interesting because Japan is in the East. However, I need to verify if Japan’s legal framework is the same as Western countries. The third and fourth results focus on religious views. The fourth result from slife.org details that in Christianity, especially Catholicism and Protestantism, euthanasia is often opposed, but some Christian groups support it. Hinduism and Buddhism have nuanced views, with some supporting it as compassion and others opposing it. Islam and Judaism also have specific stances, with Islam generally prohibiting it. This is important because religion plays a significant role in Eastern cultures, whereas Western countries might be more secular. Now, considering the East, countries like Japan, India, and others in Asia might have different perspectives. The fourth result mentions that in Japan, euthanasia is legal but under strict conditions. However, I should check if this is accurate. The Wikipedia entry for Japan in euthanasia law might have more details. Also, in the East, collectivist values and family influence might affect euthanasia decisions, unlike the individualistic approach in the West. The fifth result from The Richest includes Japan in the list of countries where euthanasia is legal, but I need to confirm the legal framework there. Japan’s legal system might have different criteria compared to Western countries. Additionally, in many Eastern countries, religious beliefs like Buddhism and Hinduism emphasize non-harm and the sanctity of life, which could lead to opposition to euthanasia. However, there might be cultural variations within the East. I should also consider the role of legal systems. Western countries with legalized euthanasia often have strict regulations and require patient consent, whereas in the East, legal frameworks might be more restrictive or non-existent. For example, in India, euthanasia is a contentious issue with legal challenges, while in Japan, it’s legal but with specific conditions. To summarize, the West tends to have more liberal legal frameworks influenced by secularism and individual rights, while the East may have more conservative views influenced by religious doctrines and collectivist values. However, there are exceptions, like Japan’s legal euthanasia, which might be a unique case. I need to structure this into a report, citing the sources provided and ensuring accurate representation of both regions’ perspectives. </think>

tool_calls: [

{

"id": "chatcmpl-tool-50852a9a85d242edbe567f61aa66c649", "function": {

"arguments": "{"url": "https://en.m.wikipedia.org/wiki/Legality_of_euthanasia"}", "name": "web_fetch"

}, "type": "function"

} ]

Role: tool tool_call_id: chatcmpl-tool-50852a9a85d242edbe567f61aa66c649 name: web_fetch content: Fetched content from https://en.m.wikipedia.org/wiki/Legality_of_euthanasia: Legality of euthanasia - Wikipedia Legality of euthanasia Language Watch Edit This article is about the legality of euthanasia of humans. For mercy killings performed on animals, see Animal euthanasia. This article includes inline citations, but they are not properly formatted. Please improve this article by correcting them. ( August 2019) ( Learn how and when to remove this template message)

Efforts to change government policies on euthanasia of humans in the 20th and 21st centuries have met limited success in Western countries. Human euthanasia policies have also been developed by a variety of NGOs, most notably medical associations and advocacy organizations. As of March 2021 update, active human euthanasia is legal in the Netherlands, Belgium, Colombia, Luxembourg, Western Australia, Canada, and Spain. Assisted suicide is legal in Switzerland, Germany, the Australian state of Victoria and in the U.S. states of Washington, Oregon, Colorado, Hawaii, Vermont, Maine, New Jersey, California, and in the District of Columbia...

Role: assistant content: <think> Okay, let’s start by understanding the user’s query about how views on euthanasia vary between the West and the East. The user wants a detailed report, so I need to cover legal, cultural, and religious aspects in both regions. First, I’ll look at the Western perspective. From the search results, countries like the Netherlands, Belgium, Canada, and some U.S. states have legalized euthanasia or physician-assisted suicide. The legal frameworks are strict, requiring patient consent and terminal illness. The Wikipedia entry mentions that these laws were enacted after significant legal battles and public debate. Culturally, the West emphasizes individual autonomy and personal rights, which supports the legalization of euthanasia. Religiously, while Christianity is predominant, there’s a trend towards secularism in some Western countries, which might contribute to more liberal policies. However, religious groups like the Catholic Church often oppose euthanasia, which could explain some resistance in certain areas. Now, for the Eastern perspective. The search results mention Japan’s unique legal framework where both passive and active euthanasia are possible under strict conditions. However, in many Eastern countries, euthanasia remains illegal or highly restricted. Cultural factors here might include collectivist values, where family and community play a significant role in decision-making. Religions like Buddhism, Hinduism, and Islam have varying views. For example, Buddhism emphasizes non-harm, which could lead to opposition against euthanasia. Hinduism’s concept of karma and dharma might also influence perspectives, as interfering with natural death could be seen as disrupting one’s spiritual journey. Islam generally prohibits euthanasia due to the belief that life is sacred and only God can determine the time of death. I should also consider the role of legal systems. Western countries with legalized euthanasia have specific laws and regulations, while Eastern countries often lack such legal frameworks. This could be due to differing legal traditions and the influence of religious doctrines on legislation. Another point is the difference in public opinion. The Wikipedia entry mentions that in some Western countries, there’s a majority support for legalizing euthanasia, whereas in the East, public opinion might be more divided due to cultural and religious influences. I need to structure the report to compare these aspects clearly. Start with an introduction, then break down the Western and Eastern perspectives into legal, cultural, and religious sections. Use citations from the provided sources, like the Wikipedia entry and the Factual Facts list, to back up each point. Make sure to highlight the key differences and any exceptions, like Japan’s legal stance in the East. I should also check if there are any recent developments or trends. For example, in the East, some countries are starting to debate euthanasia more openly, especially with aging populations. However, the report should focus on the main differences as per the available data. Finally, conclude by summarizing the main points and emphasizing the interplay between legal, cultural, and religious factors in shaping euthanasia views in both regions. Make sure all citations are properly formatted in markdown as per the user’s instructions. </think> <report> # Views on Euthanasia: A Comparative Analysis of Western and Eastern Perspectives ## Introduction Euthanasia remains a contentious issue globally, with significant variations in legal frameworks, cultural attitudes, and religious influences between the West and the East. This report explores these differences, focusing on legal status, cultural values, and religious doctrines that shape perspectives in both regions.

## Western Perspectives on Euthanasia ### Legal Frameworks Western countries exhibit a spectrum of legal approaches to euthanasia:

- - **Legalization**: Euthanasia and physician-assisted suicide (PAS) are legal in the **Netherlands, Belgium, Canada, and Spain** under strict conditions, such as terminal illness and patient consent ([Legality of Euthanasia](https://en.m.wikipedia.org/wiki/Legality_of_euthanasia)).
- - **Regulated PAS**: In the **United States**, states like Oregon, Washington, and California permit PAS for terminally ill patients ([10 Countries Where Euthanasia is Legal](https://www.therichest.com/most-influential/10-countries-where-euthanasia-andassisted-suicide-are-legal/)).
- - **Restrictions**: Many European countries, including Germany and the UK, prohibit active euthanasia but allow passive euthanasia (withdrawing life support) under specific circumstances. ### Cultural and Ethical Considerations
- - **Individual Autonomy**: Western societies often prioritize personal choice and self-determination. The concept of death¨ with dignityïs central to debates, emphasizing patients’ rights to control their end-of-life decisions.

- - **Secularism**: In countries like the Netherlands and Belgium, secular legal systems have decoupled euthanasia policies from religious doctrines, enabling progressive legislation. ### Religious Influences
- - **Christianity**: While some Christian denominations (e.g., liberal Protestant groups) support euthanasia as an act of compassion, others, like the Catholic Church, oppose it on the grounds of the sanctity of life.
- - **Slippery Slope Debates**: Critics in the West often cite ethical concerns about potential abuses, such as vulnerable populations being pressured into euthanasia.

## Eastern Perspectives on Euthanasia ### Legal Frameworks Eastern countries largely maintain restrictive or absent legal frameworks for euthanasia:

- - **Japan**: Both passive and active euthanasia are legally permissible under strict judicial guidelines, but cases remain rare. The 1995 Tokai University case established conditions requiring patient consent and exhaustion of palliative care ([Legality of Euthanasia](https://en.m.wikipedia.org/wiki/Legality_of_euthanasia)).
- - **India and South Korea**: Passive euthanasia is legal, but active euthanasia remains prohibited. For example, India’s Supreme Court legalized passive euthanasia in 2018 for patients in permanent vegetative states.
- - **Middle East and South Asia**: Countries like Saudi Arabia, Pakistan, and Indonesia prohibit euthanasia under Islamic law (Sharia), which views life as sacred and only terminable by God. ### Cultural and Ethical Considerations
- - **Collectivist Values**: In many Eastern cultures, family and community consensus play a critical role in end-of-life decisions. For instance, in Japan, bioethics¨ SWAT teamsmediate¨ disputes between patients, families, and medical professionals.
- - **Dignity and Harmony**: Concepts of social harmony and filial piety in East Asian cultures often discourage individualistic decisions that could disrupt family dynamics. ### Religious Influences
- - **Buddhism and Hinduism**:
- - **Buddhism**: Emphasizes non-harm (*ahimsa*), leading to opposition against euthanasia as an act of killing.
- - **Hinduism**: Views death as part of *karma* and *dharma*, with euthanasia potentially disrupting spiritual rebirth cycles.
- - **Islam**: Explicitly prohibits euthanasia, considering it a sin and a violation of divine will ([Religious Views on Euthanasia](https://slife.org/religious-views-on-euthanasia/)).

## Comparative Analysis | **Aspect** | **West** | **East** | |————————–|—————————————————————————|—————————————————————————| | **Legal Status** | Legal in several countries (e.g., Netherlands, Canada) with strict criteria. | Mostly illegal; Japan and India permit passive euthanasia under limited conditions. | | **Cultural Priorities** | Individual autonomy and self-determination. | Family consensus and social harmony. | | **Religious Influence** | Mixed; secularism often overrides religious opposition in Western Europe. | Strong religious prohibitions (e.g., Islam, Buddhism) shape laws and norms. | | **Public Opinion** | Generally supportive in countries with legalized euthanasia (e.g., 65% in New Zealand). | More divided due to religious and cultural norms; debates are emerging in Japan and South Korea. |

## Recent Trends and Controversies

- - **Western Expansion**: Spain legalized euthanasia in 2021, reflecting growing acceptance in Europe.
- - **Eastern Debates**: Japan’s Well-Dying¨ Bill¨ (2018) and South Korea’s discussions on end-of-life care highlight shifting attitudes in the East, though progress remains slow due to religious and cultural resistance.

## Conclusion Western and Eastern views on euthanasia diverge significantly due to legal traditions, cultural values, and religious doctrines. The West emphasizes individual rights and secular governance, enabling legalization in several countries. In contrast, the East prioritizes collectivist ethics and religious principles, resulting in stricter prohibitions. As global populations age, these differences may spark further dialogue, though deeply rooted cultural and religious frameworks will likely persist in shaping distinct perspectives. </report>

#### Task Completeness: 40.0 Citation Recall: 0.25 Citation Precision: 0.25

#### Table 10: Case 2, executed by the Qwen3-32B (Thinking) using a search tool built on a dense index.

Qid: 2024-32912 Model: Qwen3-32B (Thinking) Question: how bad did the vietnam war devastate the economy in 1968 Role: assistant content: <think> Okay, the user is asking how bad the Vietnam War devastated the economy in 1968. I need to generate a report answering this. Let me start by understanding what aspects of the economy were affected. The Vietnam War was a major conflict, so likely impacts include military spending, inflation, GDP, employment, and maybe social programs. First, I should use the web_search function to find relevant information. The query should be something like "economic impact of Vietnam War 1968". Let me check if I need to specify the year. The user mentioned 1968, so focusing on that year is important. Maybe search for "Vietnam War 1968 economic effects" or "1968 Vietnam War economy statistics". I’ll call the web_search function with the query ëconomic impact of Vietnam War in 1968änd num_results set to 5 to get a few sources. After getting the search results, I can look at the URLs and titles to see which ones are most relevant. Then, if necessary, use web_fetch to get the content from those URLs for more detailed info. Wait, but the user wants the report to include citations from the external tools. So after searching, I need to note the sources. For example, if a search result is from a government report or a reputable economic analysis site, that would be good. Let me make sure to capture the titles and URLs for citations later. Once I have the data, I can structure the report into sections: military spending, inflation, GDP growth, employment, and social programs. Each section should have statistics from 1968, supported by the sources. For example, in 1968, the U.S. spent X billion on the war, leading to Y% inflation. GDP growth was Z%, etc. I need to verify the numbers from multiple sources if possible. Also, check if there were any specific events in 1968, like the Tet Offensive, that had economic repercussions. Maybe the draft and troop numbers affected the labor force. Additionally, the Vietnam War’s impact on public spending versus social programs like the Great Society might be relevant. After gathering all the information, I’ll compile it into a markdown report with headings, bullet points for key stats, and citations at the end of each relevant sentence. Make sure each citation is properly formatted in brackets with title and URL. Avoid listing all sources at the end; instead, integrate them where the info is used. Let me start by making the initial web_search call. If the results are not sufficient, I might need to do another search with adjusted keywords. Once I have enough data, I can proceed to write the report. </think> tool_calls: [

{

"id": "chatcmpl-tool-4bb107a4d963440fa061839f1dccbccd", "function": {

"arguments": ""query": "economic impact of Vietnam War in 1968", "num_results": 5", "name": "web_search"

}, "type": "function"

} ]

Role: tool tool_call_id: chatcmpl-tool-96da884e41f64f688161ba1bcdb99242 name: web_search content:

[

{

"title": "American Economy - How Did the Vietnam War Affect America?", "headings": "How Did the Vietnam War Affect America?", "url": "https://vietnamwaraffects.weebly.com/american-economy.html"

}, {

"title": "Vietnam War and the Economy", "headings": "Vietnam War and the Economy", "url": "https://www.historycentral.com/sixty/Economics/Vietnam.html"

}, {

"title": "Vietnam’s economy after the war (1975-1986), Vietnam Economy | Vietnam Tours", "headings": "Vietnam’s economy after the war (1975-1986); Vietnam’s economy after the war (1975-1986);

Vietnam’s economy in the war; Vietnam’s economy since reform in 1986", "url": "https://www.toursinvietnam.com/vietnam-travel-guide/Vietnams-economy-after-the-war-1975-

1986.html" }, {

"title": "1968 in the Vietnam War - Wikipedia", "headings": "1968 in the Vietnam War 1968 in the Vietnam War", "url": "https://en.wikipedia.org/wiki/1968_in_the_Vietnam_War"

}, {

"title": "How much did the Vietnam War cost? - Answers", "headings": "How much did the Vietnam War cost? How much did the Vietnam War cost? Your Answer How

much did the Vietnam War cost? ",

"url": "http://www.answers.com/Q/How_much_did_the_Vietnam_War_cost" } ]

Role: assistant content: <think> Okay, let’s start by analyzing the user’s question: how¨ bad did the Vietnam War devastate the economy in 1968.The¨ user is asking about the economic impact of the Vietnam War specifically in 1968. First, I need to recall the general economic effects of the Vietnam War. I know that the U.S. spent a significant amount on the war, which contributed to inflation and budget deficits. However, the user is focusing on 1968, so I need to find specific data from that year. Looking at the search results provided: 1. The first result from VietnamWarAffects.weebly.com discusses the American economy’s impact from the Vietnam War. It mentions that the war led to increased government spending, inflation, and a decline in public services. However, the exact figures for 1968 are not provided here. 2. The second result from HistoryCentral.com’s Vietnam¨ War and the Economypage¨ likely details the economic consequences. It might include statistics on military expenditure, GDP, and inflation rates in 1968. 3. The third result is about Vietnam’s economy post-1975, which is outside the user’s scope of 1968, so it’s probably not relevant here. 4. The fourth result is the Wikipedia page for 1968 in the Vietnam War. While it provides a timeline of events, it may not focus on economic data. However, it could mention key events in 1968 that had economic repercussions, such as the Tet Offensive. 5. The fifth result from Answers.com asks about the total cost of the Vietnam War but doesn’t specify 1968.

The answer might give an overall figure, but I need to check if it breaks down the costs by year. Since the user wants information specifically for 1968, I need to extract data from these sources that pertain to that year. The first and second results seem most relevant. The Wikipedia page might have some economic context within the events of 1968, like increased military spending or draft impacts. I should check if the HistoryCentral page provides specific numbers for 1968. For example, how much the U.S. spent on the war in that year, the inflation rate, and GDP growth. The Weebly page might mention the overall effects but not the exact 1968 figures. The Answers.com page might have total costs, but I need to see if it can be used to estimate the 1968 expenditure. If the total cost is known, I could look for the percentage allocated in 1968. However, without specific data, this might be an approximation. The Wikipedia page for 1968 in the Vietnam War might mention the number of troops, which could relate to economic impact through military spending and the draft. For instance, the U.S. had around 500,000 troops in Vietnam by 1968, which would require significant resources. I also need to consider the broader economic context of 1968 in the U.S., such as the Great Society programs being affected by the war’s funding. The war’s cost may have led to cuts in social programs or increased taxes. Potential sources for specific data: - The HistoryCentral page might state that in 1968, the U.S. spent approximately $20 billion on the war, leading to a budget deficit. This could be a key figure to include. - The Weebly page might discuss how the war contributed to inflation. In 1968, the inflation rate in the U.S. was around 4.2%, which could be linked to war spending. - The Wikipedia page might note that the war effort led to a recession in 1968-1969, with rising unemployment and stagnant economic growth. I should verify these points by checking the content of the provided search results. Since I can’t access the full content, I’ll have to infer based on the titles and headings. For example, the HistoryCentral page’s title Vietnam¨ War and the Economysuggests¨ it will cover economic impacts, so it’s likely to have the necessary statistics. Putting this together, the report should outline the military expenditure in 1968, the resulting inflation, the impact on GDP, and any related economic policies or recessions. Each point should be supported by citations from the relevant sources. </think> <report> # Economic Impact of the Vietnam War in 1968 The Vietnam War had profound economic consequences for the United States in 1968, characterized by massive military spending, rising inflation, and strained public finances. Below is a detailed analysis of these effects, supported by historical data and sources.

- ## 1. **Military Spending and Federal Budget Deficits** In 1968, the U.S. allocated approximately **$20 billion** (equivalent to over $150 billion in 2023) to fund military operations in Vietnam. This expenditure accounted for nearly **5% of the federal budget** and contributed to a **$2.3 billion budget deficit** for the year ([Vietnam War and the Economy](https://www.historycentral.com/sixty/Economics/Vietnam.html)). The war’s financial burden forced cuts in domestic programs and redirected resources from social initiatives like the Great Society to war efforts.
- ## 2. **Inflation and Monetary Policy** The war exacerbated inflation, which reached **4.2% in 1968**, up from 2.9% in 1967. The Federal Reserve struggled to balance anti-inflationary measures with the need to fund the war, leading to rising interest rates. The combination of increased government spending and inflation eroded purchasing power, particularly for middle- and lower-income households ([American Economy - How Did the Vietnam War Affect America?](https://vietnamwaraffects.weebly.com/american-economy.html)).
- ## 3. **Economic Recession and Unemployment** By late 1968, the U.S. entered a recession marked by **rising unemployment** (4.2% in 1968, peaking at 3.8% in 1969) and stagnant GDP growth. The war’s demand for resources and labor diverted investment from private sectors, while draft-related labor shortages further strained the economy. The recession was compounded by public discontent and political instability ([1968 in the Vietnam War - Wikipedia](https://en.wikipedia.org/wiki/1968_in_the_Vietnam_War)).
- ## 4. **Social and Long-Term Costs** The war’s indirect costs included **widespread social unrest** and long-term economic consequences. The 1968 Tet Offensive, a pivotal military event that year, intensified public opposition and led to calls for withdrawal. Additionally, the war’s toll on infrastructure, healthcare, and veterans’ benefits persisted for decades ([How much did the Vietnam War cost? Answers](http://www.answers.com/Q/How_much_did_the_Vietnam_War_cost)). ## Conclusion The Vietnam War’s economic impact in 1968 was severe, with immediate effects on inflation, public spending, and employment. These challenges set the stage for broader economic crises in the 1970s, including the 1973 oil shock and stagflation. The war underscored the difficulty of sustaining large-scale military conflicts without compromising domestic economic stability. </report> Task Completeness: 26.9 Citation Recall: 0.0 Citation Precision: 0.0

