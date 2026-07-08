## arXiv:2510.05396v2[cs.IR]8Oct2025

[Figure 1]

2025-10-09

# Scalable In-context Ranking with Generative Models

###### Nilesh Gupta1, Chong You3, Srinadh Bhojanapalli3, Sanjiv Kumar3, Inderjit Dhillon1,2 and Felix Yu3

1UT Austin, 2Google, 3Google DeepMind

In-context Ranking (ICR) is an emerging paradigm for Information Retrieval (IR), which leverages contextual understanding of LLMs by directly incorporating the task description, candidate documents, and the query into the model’s input prompt and tasking the LLM to identify relevant document(s). While it is effective, efficiency is a significant challenge in this paradigm, especially as the candidate list grows due to quadratic / super-linear scaling of attention operation with context length. To this end, this paper first identifies inherent and exploitable structures in the attention of LLMs finetuned for ICR: (1) inter-document block sparsity – attention is dense within each document block but sparse across different documents in the context; and (2) query-document block relevance – the attention scores from certain query tokens to a document block in middle layers strongly correlate with that document’s actual relevance. Motivated by these observations, we introduce BlockRank (Blockwise In-context Ranking), a novel method that adapts the attention operation in an LLM by (a) architecturally enforcing the observed inter-document block sparsity, reducing attention complexity from quadratic to linear without loss in performance, and (b) optimizing query-document block relevance for true relevant documents during fine-tuning using an auxiliary contrastive training objective, improving retrieval in attention. Experiments on BEIR, MSMarco and NQ with Mistral-7B demonstrate that BlockRank Mistral matches or outperforms existing SOTA listwise rankers and controlled fine-tuned baseline while being significantly more efficient at inference (4.7× for 100 MSMarco documents in context) and scaling gracefully to long-context shortlists - around 500 documents in-context (∼ 100𝐾 context length) within a second, presenting a scalable and effective solution for ICR. We will make our code available here.

[Figure 2]

[Figure 3]

[Figure 4]

Figure 1 | Analysis of attention patterns in Mistral-7B performing In-context Ranking (ICR) on MSMarco. (left) Attention averaged over middle layers 16-21 reveals structural sparsity — a strong diagonal (intra-document attention needed for local context processing) and significant attention to the first row (focus on the query-based instruction). (middle) Attention in Layer 18 from individual query tokens to document segments. Certain tokens (the last token, ‘:’) attend primarily to the relevant document only (i.e., Doc24, highlighted in green). (right) Attention from final query tokens across layers shows retrieval signals strengthening in middle layers. These patterns motivate our BlockRank approach.

Corresponding author(s): nilesh@cs.utexas.edu (Work done during internship at Google), cyou@google.com

### 1. Introduction

Information retrieval (IR) is the problem of finding relevant content from a large document corpora. While sparse retrieval methods based on word-level matching have existed for decades (Formal et al., 2021; Robertson et al., 2009), modern IR systems increasingly leverage deep neutral network based representations, which achieve their success through a superior ability to capture deep semantic relationships (Karpukhin et al., 2020). Recently, generative large language models (LLMs) (Achiam et al., 2023; Team et al., 2023) have emerged as a revolutionary paradigm that transforms many sub-fields of machine learning, including IR. Through pre-training on the web, LLMs absorb an enormous amount of world knowledge and demonstrate remarkable capabilities in dialogue, question answering, reasoning, and beyond (Wei et al., 2022).

The powerful capabilities of LLM open up novel approaches for IR as well. One emerging paradigm is the In-context Ranking (ICR) (Lee et al., 2024; Ma et al., 2023), which directly leverages an LLM’s contextual understanding capabilities. In this setup, a query and a list of candidate documents are formatted together within the LLM’s input prompt (see Figure 3), tasking the model to identify the most relevant document(s), often through the generative decoding process. ICR holds the promise of considering the query and all candidates simultaneously while performing relevance judgements.

Despite this promise, LLM-based ICR introduces significant efficiency challenges. As the number of candidate documents increases, the input context length grows rapidly, making inference computationally expensive and memory-intensive, due to quadratic/super-linear complexity of the attention mechanism. Current methods (Lee et al., 2024; Pradeep et al., 2023a,b; Sun et al., 2023) typically treat the LLM as a black-box or do not fully utilize the structure of the ICR task i.e. the input prompt is composed of a sequence of potentially independent candidate documents conditioned on a shared instruction prompt. Moreover, as we discuss in Section 5.3 and Section D.1 of Appendix, auto-regressive decoding is not best suited for this task when decoding multiple predictions from the fine-tuned model (see Table 4).

Paper Contributions. To this end, we first investigate how standard LLMs process information within the specific task structure. We conduct an analysis of the attention patterns of a fine-tuned Mistral-7B model when prompted on ICR examples derived from MSMarco dataset (see Section 3 for details and Figure 1 for visualizations). This analysis reveals two structural properties: (1) inter-document block sparsity – most document tokens focusing locally (primarily within their own document, on instructions, or one or two other documents), rather than attending densely across all candidate documents. (2) query-document block relevance – similar to the findings of Chen et al. (2025); Wu et al. (2024), we find that last and some specific query tokens like “:” (that signal start of the potential document generation process) develop strong attention weights towards relevant document tokens, particularly in the model’s middle layers.

Building up on these insights, we propose BlockRank (Blockwise In-context Ranking), an efficient incontext ranking method. BlockRank introduces two modifications (visualized in Figure 2) to standard LLM architecture and fine-tuning: (1) architecturally, it imposes a structured sparse attention in which document tokens attend only causally to their own content and shared instruction tokens, reducing attention complexity from quadratic to linear; and (2) it incorporates a contrastive learning objective to explicitly optimize internal attention from signal-carrying query tokens toward relevant documents, which helps the BlockRank model in two fronts: (a) attend strongly to the relevant document in context, improving retrieval quality (see Table 3); (b) the ability to reliably infer relevance based on attention concentration during the prefill stage, leading to further speedups in inference compared to iterative decoding (see Section 4.3). To summarize, the main contributions of this work are:

Standard Fine-tuning BlockRank Fine-tuning

Info NCE

Unembedding layer

NTP

Full Prompt

Prompt segmentation & Chunking

Layer L

Attention Relevance Scores

You'll be given a list..

Inst

- ID: 1 | ... | END ID: 1
- ID: 2 | ... | END ID: 2

Doc2

... ID: i* | ... | END ID: i* ... ===Now let's start=== Which doc... query: ... Final Answer: [

Layer l*

|Doci*|
|---|

|Query|
|---|

Structured Attention

Embedding Layer

- Figure 2 | BlockRank starts with chunking the full prompt into segments and then processes it using structured attention, where the documents only attend to themselves and the instruction segment, while the query segment attends to the full prompt. It also incorporates an auxiliary attention loss

(Laux) from a middle layer (𝑙∗) that increases sharpness of attention on the relevant documents and enables an alternate inference mechanism using attention scores derived from 𝑙∗.

- • an analysis that characterizes attention patterns in LLMs fine-tuned for ICR, identifying key sparsity structures and latent retrieval signal carriers (specific query tokens in middle layers).
- • an efficient approach BlockRank for In-context Ranking that enforces a structured sparse attention and a contrastive training objective on internal attention.
- • extensive experiments on standard retrieval benchmarks (BEIR, MSMarco and NQ) demonstrating that BlockRank achieve strong ICR performance, matching or outperforming strong baselines as well as full fine-tuned model (see Table 1, 2), while being order of magnitude efficient at inference (see Figure 4).

The remainder of this paper is organized as follows: Section 2 describes the problem setup and discusses related work. Section 3 details our analysis of LLM attention in ICR. Section 4 presents the BlockRank methodology. Section 5 reports experimental results, and Section 6 concludes the paper.

### 2. Problem Setup and Related Work

This section formally defines the ICR task addressed in this paper. We also review relevant prior work that uses LLM for IR, and position our BlockRank method in the context of the literature.

###### 2.1. Problem Formulation: In-context Ranking

Given a collection of 𝑛 documents D = {𝑑1, . . . , 𝑑𝑛} and a query 𝑞, the goal of IR is to return a subset of D that are relevant to 𝑞. In this paper, we consider documents and queries in the form of text that can be parsed by an LLM, though the discussion may also apply to visual and audio data when the LLM is multi-modal. Following standard practices (Lee et al., 2024), we define an ICR prompt as a composition of the list of documents together with the query as following:

prompt(𝑞, D) “{Inst}. {𝑑1}, . . . , {𝑑𝑁}. {𝑞}” (1)

In practice, processing the entire corpus D (where 𝑛 can be millions) within a single prompt is infeasible due to LLM context length limitations. Therefore, the ICR task we consider in this paper operates on a smaller candidate list D𝑞 = (𝑑1, . . . , 𝑑𝑁) ⊆ D, where 𝑁 is the number of candidates (𝑁 = O(100) in our experiments) retrieved by a first-stage retrieval model (e.g. dual-encoder). The

You will be given a query and a list of documents. Each document will be formatted as ID: <id> | CONTENT: <content> | END ID: <id>. You need to read carefully and understand all of them. The query is: which classification group contains the most organisms, and your goal is to find all document(s) from the list that can help answer the query.

ID: 1 | CONTENT: This is a diverse group of organisms. It includes plants, animals.. | END ID: 1

... [Documents in-between omitted for brevity] ...

ID: 𝑁 | CONTENT: Organisms composed of eukaryotic cells are divided into 4 main.. | END ID: 𝑁

====== Now let’s start! ====== Which document is most relevant to answer the query? Print out the ID of the document. Query: which classification group contains the most organisms. The following document(s) can help answer the query:

Final Answer: [‘20’]

- Figure 3 | Example structure of the prompt template used in our experiments, showing query-based instruction, abbreviated document list, and the final query section.

prompt in (1) is thus applied to this candidate list D𝑞. Furthermore, each document representation {𝑑𝑖} within the prompt often includes structured formatting beyond just the raw text 𝑐𝑖, such as its unique identifier 𝑖𝑑𝑖. We adopt the format from (Lee et al., 2024), also illustrated in Figure 3, explicitly demarcating document start, content, and end, along with identifiers (e.g., ID: 𝑖𝑑𝑖 | CONTENT: 𝑐𝑖 | END ID: 𝑖𝑑𝑖).

Inst is a description of the retrieval task and can also include the query 𝑞. While excluding the query from the instruction prefix is desirable from an efficiency standpoint – as it would allow for the query-independent representations of documents to be pre-processed and cached offline – we find this leads to a noticable drop in performance in our experiments (see Table 6). We hypothesize, including the query in Inst allows the model to condition each document’s representation on the specific information need from the outset, enabling it to better focus on query-relevant facts and signals within each document during processing. We note that existing listwise LLM re-rankers (Pradeep et al., 2023b; Sun et al., 2023) also apply a similar formatting where the query appears before the documents. Our preliminary experiments show that one can replace the query with a similar-looking document from the corpus, suggesting that future work can potentially explore conditioning document representations within clusters to alleviate the need for query-dependent processing.

The objective of In-context Ranking is then formally defined as follows: given the prompt(𝑞, D𝑞) constructed from the query 𝑞 and the candidate list D𝑞, train or utilize an LLM 𝑓𝜃, with 𝜃 being the model weight, to effectively identify and output the identifiers 𝑖𝑑∗ corresponding to the 𝑑∗ ∈ D𝑞 deemed relevant to 𝑞. Typically it is achieved by predicting a ranked permutation of D𝑞 and taking the top 𝑘 elements. The central challenge addressed in this paper is to develop methods to train 𝑓𝜃 and perform this prediction both effectively (high accuracy) and efficiently (low computational cost), particularly as the candidate list size 𝑁 increases.

###### 2.2. Related Work

Our work builds upon research in neural retrieval, the rapidly evolving field of using LLMs for IR, and efficient attention mechanisms.

Neural Re-ranking and Retrieval Models. Prior to LLMs, neural information retrieval saw significant progress with methods like dense dual-encoder retrieval (e.g., DPR (Karpukhin et al., 2020), ANCE (Xiong et al., 2020)) offering efficient first-stage filtering, and cross-encoder models (e.g., monoBERT (Nogueira and Cho, 2020), monoT5 (Nogueira et al., 2020)) providing high re-ranking effectiveness through deep query-document interaction. Late interaction models like ColBERTv2 (Santhanam et al., 2022) aimed to balance the trade-off between dual and cross encoders. Our work is architecturally distinct from traditional neural IR methods as it operates within the in-context ranking paradigm, where a single LLM processes the query and the entire candidate list simultaneously in one context window allowing full contextualization (query and output representations are conditioned on the full set of candidate documents) and complex instruction-following (e.g., "Find documents that disagree with...").

LLMs as Listwise Re-rankers. The ability of LLMs to process and reason over long contexts spurred their application to listwise re-ranking (Ma et al., 2023) (or In-context Ranking), where multiple candidates are processed simultaneously. Initial successes often involved prompting large proprietary models like GPT-3.5/4 (Sun et al., 2023) in zero-shot or few-shot settings. While effective, these approaches typically incur high computational costs and often rely on auto-regressive generation to output rankings or relevance scores, adding latency. More recent work focuses on adapting opensource LLMs (e.g., Llama, Mistral, Zephyr, Vicuna) for this task (Pradeep et al., 2023a,b; Zhang et al.,

- 2023) and improving efficiency, for instance using Seq2Seq architectures (Tamber et al., 2023) and using single-token decoding (Reddy et al., 2024). Recent papers have also shown the existence of retrieval heads (attention heads that carry strong retrieval signals) in many modern LLMs (Wu et al.,
- 2024) and their usefulness in inferring retrieval signals (Chen et al., 2025). Our work differs from these methods by introducing effective task specific restructuring of the attention architecture for efficiency and an explicit fine-tuning objective to directly train the model’s attention patterns for the ranking task.

In-context Retrieval / Ranking Lee et al. (2024) studied the In-context Retrieval (ICR) paradigm for various frontier LLMs, demonstrating that long-context models can match the performance of specialized retrieval systems when processing corpora of up to a few thousand documents. Our work builds on this paradigm but addresses a more challenging setting. The evaluation in that study is performed on a random subset of documents from the full corpus. In contrast, our experiments focus on ranking the top-k hard candidates returned by a strong first-stage retriever i.e. In-context Ranking task. This task is arguably more difficult, as the model must distinguish between many semantically similar documents to identify the correct answer. We argue that this hard negative setting is a more faithful simulation of the practical application of LLMs in retrieval pipelines. Moreover, processing long lists of documents within the LLM context remains challenging, with studies highlighting difficulties in effectively utilizing long-range information (Goldman et al., 2024).

Efficient and Structured Attention. The quadratic complexity of the standard self-attention has spurred extensive research into more efficient attention approximations. Many successful approaches enforce a structured sparsity on the attention matrix, reducing complexity from quadratic to sublinear. Notable examples include methods based on sliding windows (e.g., Longformer (Beltagy et al., 2020)), global-local patterns (e.g., BigBird (Zaheer et al., 2020)), and other block-wise structures. While these methods are designed for general long-context processing, BlockRank’s structured attention can be seen as a task-specific instance of this paradigm. BlockRank’s sparsity is semantically informed by the logical structure of the in-context ranking task itself—separating instructions, documents, and the query. This content-aware structuring allows for a highly efficient architecture tailored for ICR.

### 3. Emergence of Structured Attention in In-context Ranking

Before introducing our method, we first analyze the attention mechanisms of a standard LLM when performing the ICR task defined in Section 2.1. The analysis below is anchored on Figure 1 which is based on a random sample, we provide more results in the Appendix Section D.

Analysis Setup. We conduct our analysis using a Mistral-7B-v0.3 model (Jiang et al., 2023) finetuned on the ICR task with data derived from MSMarco (as described in Section 5.1). In particular, our fine-tuning objective is the standard Next Token Prediction (NTP) loss, without any modifications. We feed this model prompt(𝑞, D𝑞). Let the resulting input token sequence be 𝑇 = (𝑡1, . . . , 𝑡𝐿).

Our analysis focuses on the attention probabilities computed within the transformer layers. Given a layer index 𝑙 ∈ {1, . . . , 𝐿𝑚𝑜𝑑𝑒𝑙} where 𝐿𝑚𝑜𝑑𝑒𝑙 is the total number of layers, and an attention head index ℎ, we denote the attention probability from a query token 𝑡𝑖 to a key/value token 𝑡𝑗 as 𝛼𝑖𝑗(𝑙,ℎ). We often consider the attention averaged across all 𝐻 heads in a layer: 𝛼𝑖𝑗(𝑙) = 𝐻1 ℎ 𝐻=1 𝛼𝑖𝑗(𝑙,ℎ). We examine interactions between different types of tokens by partitioning the token indices {1, . . . , 𝐿} into sets corresponding to the instructions (𝑇𝐼𝑛𝑠𝑡), the query (𝑇𝑞), and each document (𝑇𝑑𝑘 for 𝑘 ∈ {1, . . . , 𝑁}). We visualize these interactions using heatmaps, with representative examples shown in Figure 1.

- Observation 1: Inter-document Block Sparsity Our first key observation is that the attention patterns exhibited by document tokens are structured and sparse, rather than uniformly dense. This is clearly visible in Figure 1(a), which shows the segment-wise attention in the middle layers. The heatmap is dominated by the diagonal, indicating strong intra-document attention: for a token 𝑡𝑖 ∈ 𝑇𝑑𝑘,

the sum of attention probabilities towards other tokens within the same document, 𝑡

𝑗∈𝑇𝑑𝑘 𝛼𝑖𝑗(𝑙), is significantly higher than attention towards other parts of the context.

This observed structured sparsity implies that computing full attention matrix might be largely redundant for this task. A significant portion of the computation could potentially be saved by enforcing an attention pattern that focuses on local (intra-document) and instructional context, directly motivating the structured sparse attention employed in BlockRank.

- Observation 2: Query-document Block Relevance Our second key observation is that certain

tokens within the query 𝑇𝑞 attends primarily to relevant documents only, particularly in the middle layers of the transformer.

Figure 1(b) illustrates this at Layer 18. It maps the attention from individual query tokens (x-axis) to the different document segments (y-axis). We observe that certain tokens, such as delimiters (‘:’) and end of prompt tokens, exhibit distinct sharp attention distributions. These specific “signal carrier” tokens attend more strongly towards the segment corresponding to the ground-truth relevant document 𝑑∗ (i.e., Doc24, highlighted in the figure) compared to irrelevant documents 𝑑𝑘 (𝑘 ≠ ∗). Formally, let 𝐴𝑖(→𝑙)𝑑

𝑗∈𝑇𝑑𝑘 𝛼𝑖𝑗(𝑙) be the total attention from query token 𝑡𝑖 to document 𝑑𝑘 at layer 𝑙. For specific 𝑡𝑖 ∈ 𝑇𝑞 identified as signal carriers and middle layers 𝑙, we observe 𝐴𝑖(→𝑙)𝑑∗ > 𝐴𝑖(→𝑙)𝑑

= 𝑡

𝑘

for most 𝑘 ≠ ∗. We hypothesize that such structural tokens carry strong retrieval signals as they often precede 𝑇𝑑∗ (by design during fine-tuning but also during pre-training), hence their attention gets biased towards the in-context 𝑑∗ segment in order to predict the succeeding 𝑇𝑑∗ tokens.

𝑘

Furthermore, the layer depth plays a critical role in the emergence of these signals. Figure 1(c) tracks the attention 𝐴𝑖(→𝑙)𝑑

from final query tokens 𝑡𝑖 to all document segments 𝑑𝑘 across all layers 𝑙 ∈ {1, . . . , 𝐿𝑚𝑜𝑑𝑒𝑙}. The plot shows that the discriminative signal is weak in the initial layers, emerges and strengthens significantly in the middle layers (approximately layers 8-24), and persists or slightly diffuses in the final layers.

𝑘

### 4. BlockRank: Blockwise In-context Ranking

Motivated by the attention analysis presented in Section 3, we propose Blockwise In-context Ranking (BlockRank), an efficient in-context ranking method. BlockRank comprises of following components (see Figure 2): a structured attention mechanism enforcing sparsity, an auxiliary attention loss to enhance retrieval signals in attention operation, and an alternative attention-based inference method. We detail each component below.

###### 4.1. Blockwise Structured Attention

The core of BlockRank’s efficiency during fine-tuning and inference stems from restructuring of attention mechanism designed to enforce the sparse patterns observed in Section 3.

###### Enforcing inter-document block sparsity. we modify attention operation such that:

- • Document Tokens (𝑡𝑖 ∈ 𝑇𝑑𝑘 for 𝑘 ∈ {1, . . . , 𝑁}): only attend to tokens within their own document chunk (𝑡𝑗 ∈ 𝑇𝑑𝑘) and tokens within the instruction chunk (𝑡𝑗 ∈ 𝑇𝐼𝑛𝑠𝑡).
- • Query Tokens (𝑡𝑖 ∈ 𝑇𝑞): attend to all tokens in the prompt (𝑡𝑗 ∈ 𝑇 = ∪𝑘𝑇𝑘) to gather context for identifying the relevant document(s).
- • Instruction Tokens (𝑡𝑖 ∈ 𝑇𝐼𝑛𝑠𝑡): attend causally within the instruction segment itself.

Instead of constructing large, explicit sparse attention masks, we implement this structured attention efficiently using the chunked representation defined as follows: the long prompt is first segmented into its logical components 𝑆0 = 𝐼𝑛𝑠𝑡, 𝑆𝑘 = 𝑑𝑘 for 𝑘 ∈ {1, . . . , 𝑁}, and 𝑆𝑁+1 = 𝑞. Each segment 𝑆𝑘 is then processed (via standard sequence length chunking or padding) to form fixed-length chunks, typically of length 𝐿𝑐ℎ𝑢𝑛𝑘 tokens. Let the token sequence corresponding to chunk 𝑆𝑘 be 𝑇𝑘 ⊂ 𝑇 = (𝑡1, . . . , 𝑡𝐿), where 𝑇 is the (potentially virtual) concatenation of all chunk sequences.

Each chunk 𝑆𝑘 can be processed largely in parallel (e.g., distributed along the batch dimension).

Let 𝑄𝑘(𝑙), 𝐾𝑘(𝑙),𝑉𝑘(𝑙) be the query, key, and value matrices for chunk 𝑆𝑘 at layer 𝑙. The attention output for a token 𝑡𝑖 in chunk 𝑆𝑘 is computed as follows:

- • If 𝑆𝑘 is a document chunk (𝑘 ∈ {1, . . . , 𝑁}): The attention output is computed using self-attention within the chunk and cross-attention only to the keys and values from the instruction chunk:

𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄𝑘(𝑙), [𝐾𝑘(𝑙), 𝐾𝐼𝑛𝑠𝑡(𝑙) ], [𝑉𝑘(𝑙),𝑉𝐼𝑛𝑠𝑡(𝑙) ]). Attention to other document chunks 𝑆𝑚≠𝑘 and the query chunk 𝑆𝑞 is effectively masked out.

- • If 𝑆𝑘 is the query chunk (𝑘 = 𝑁+1): The attention output is computed using self-attention within the

chunk and cross-attention to the keys and values from all other chunks: 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄𝑞(𝑙), [𝐾𝑞(𝑙), 𝐾𝐼𝑛𝑠𝑡(𝑙) , 𝐾𝑑(𝑙)

1

,

. . . , 𝐾𝑑(𝑙)

𝑁

], [𝑉𝑞(𝑙),𝑉𝐼𝑛𝑠𝑡(𝑙) ,𝑉𝑑(𝑙)

1

, . . . ,𝑉𝑑(𝑙)

𝑁

]).

- • Instruction chunk attention (𝑘 = 0) is standard causal self-attention.

This computes only the necessary attention scores, drastically reducing the computational cost, converting quadratic attention to linear. Please see Appendix Section C for more details and complexity analysis.

Permutation-invariant Position Embedding. To complement the structured attention, we employ a specialized position embedding that reinforces the logical separation of the prompt’s components. This also helps the model learn position-invariant representations for documents (Tang et al., 2023) and distinguish the query’s unique role. Specifically, tokens in the instruction segment (𝑇𝐼𝑛𝑠𝑡) are assigned standard sequential positions starting from 0. For all document segments (𝑇𝑑𝑘), we use a shared local position space. Each document’s tokens are assigned positions beginning immediately

after the instruction segment, as if it were the only document present. For example, if the instruction has length 𝐿𝐼𝑛𝑠𝑡, the first token of every document 𝑑𝑘 is assigned the position 𝐿𝐼𝑛𝑠𝑡. This encourages the model to apply a consistent, order-invariant function to each document, mitigating any bias from its absolute position in the candidate list. Finally, to distinctly separate the query from the document corpus, its tokens (𝑇𝑞) are assigned positions starting from a large, fixed offset. In our experiments, we use an offset of 8192, so the query tokens receive positions [8192, 8193, . . . ]. This large gap ensures that the relative positional encodings between any query token and any document token are significantly different.

###### 4.2. Auxiliary Attention Loss (Laux)

To explicitly optimize query-document block relevance for relevant documents during fine-tuning, we introduce an auxiliary loss Laux applied at a specific middle layer 𝑙∗ (determined empirically, see Section D.3 in the Appendix). This loss encourages “signal-carrier” query tokens to attend more strongly to the relevant document.

More specifically, let 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 ⊂ 𝑇𝑞 be the set of indices for the identified signal-carrying query tokens. Based on our prompt template and empirical analysis we set 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 = [“:”,“[‘”]. Let 𝑇𝑑𝑜𝑐𝑠 = 𝑘 𝑁=1 𝑇𝑑𝑘 be the set of indices for all tokens belonging to any document segment. For each signal token 𝑡𝑖 ∈ 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 at layer 𝑙∗, we compute attention scores towards document tokens 𝑡𝑗 ∈ 𝑇𝑑𝑜𝑐𝑠

- as following:

- 1. Obtain query vectors 𝑄(𝑙

∗)

𝑖 for 𝑡𝑖 ∈ 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 and key vectors 𝐾(𝑙

∗)

𝑗 for 𝑡𝑗 ∈ 𝑇𝑑𝑜𝑐𝑠.

- 2. Compute raw attention logits 𝑧𝑖𝑗 = 𝑄(𝑙

∗)

𝑖 (𝐾(𝑙

∗)

𝑗 )𝑇/

√𝑑𝑘 for all 𝑡𝑗 ∈ 𝑇𝑑𝑜𝑐𝑠.

- 3. Compute normalized attention probabilities only over the document tokens: 𝛼′

𝑖𝑗 = softmax𝑗(𝑧𝑖𝑗), where the softmax is computed across all 𝑗 such that 𝑡𝑗 ∈ 𝑇𝑑𝑜𝑐𝑠. This normalization focuses the probability mass exclusively on the candidate documents, ignoring instructions and query tokens.

- 4. Aggregate these probabilities to compute an attention mass score for each document 𝑑𝑘:

𝑆(𝑞, 𝑑𝑘) = 𝑡

𝑖∈𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 𝑡𝑗∈𝑇𝑑𝑘 𝛼′

𝑖𝑗 (Alternative: could use mean aggregation over 𝑡𝑖). This score 𝑆(𝑞, 𝑑𝑘) quantifies the relevance signal from the carrier tokens towards document 𝑑𝑘.

- 5. Apply a contrastive loss using these scores. We use the InfoNCE loss with temperature 𝜏:

exp(𝑆(𝑞, 𝑑∗)/𝜏)

Laux = LInfoNCE(𝑆(𝑞, 𝑑∗), {𝑆(𝑞, 𝑑𝑘)}𝑘≠∗;𝜏) = −log

(2)

𝑁 𝑘=1 exp(𝑆(𝑞, 𝑑𝑘)/𝜏)

where 𝑑∗ is the ground-truth relevant document. This loss encourages the score 𝑆(𝑞, 𝑑∗) for the relevant document to be higher than scores for irrelevant documents.

Overall Training Objective. The BlockRank model is fine-tuned by minimizing a combined loss function that includes both the standard next-token prediction objective and our auxiliary attention loss:

L𝑇𝑜𝑡𝑎𝑙 = L𝑁𝑇𝑃 + 𝜆Laux (3)

Here, L𝑁𝑇𝑃 is the cross-entropy loss calculated on the answer tokens (similar to standard instruction tuning) based on the model’s prediction of the next token in the sequence, computed using the final hidden states which are generated respecting the structured attention masks defined in Section 4.1. Laux is the auxiliary InfoNCE loss defined in Equation 2, applied only at layer 𝑙∗. 𝜆 is a hyperparameter balancing the two losses (we use 𝜆 = 0.1 in our experiments).

###### 4.3. Efficient Attention-Based Inference

An advantage of BlockRank is that the auxiliary loss explicitly optimizes the attention scores 𝑆(𝑞, 𝑑𝑘) to reflect relevance. This allows for an alternate efficient inference mechanism during the prefill stage of the context processing. It can bypass the iterative auto-regressive decoding process, and even the full forward pass (depending on the choice of 𝑙∗). The inference mechanism can be defined as follows:

- 1. Given a prompt(𝑞, 𝐷), perform a partial forward pass of the BlockRank model up to the target middle layer 𝑙∗.
- 2. Compute the document relevance scores 𝑆(𝑞, 𝑑𝑘) for all candidate documents 𝑘 ∈ {1, . . . , 𝑁} using the exact same procedure as described for the auxiliary loss calculation (Section 4.2, steps 1-4), utilizing the signal carrier tokens 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 and performing the softmax over document tokens 𝑇𝑑𝑜𝑐𝑠 only.
- 3. Identify the index 𝑘ˆ of the document with the highest score: 𝑘ˆ = argmax𝑘 𝑆(𝑞, 𝑑𝑘).
- 4. Output the corresponding document identifier 𝑖𝑑𝑘ˆ, for top-𝐾 predictions output argtop𝑘𝑆(𝑞, 𝑑𝑘)
- 5. Experimental Results

This section empirically evaluates the proposed BlockRank method. We conduct two sets of experiments: first, an evaluation on the BEIR benchmark to assess zero-shot generalization against state-of-the-art re-rankers, and second, a controlled in-domain evaluation to analyze effectiveness, efficiency, and scalability. We aim to answer the following research questions: (RQ1) How does BlockRank compare against strong baselines in terms of retrieval effectiveness, both in zero-shot generalization and in-domain settings? (RQ2) What are the efficiency benefits of BlockRank compared to standard fine-tuning, particularly when scaling the number of in-context documents? (RQ3) What is the contribution of BlockRank’s core components (structured sparse attention, auxiliary attention loss, and attention-based inference) to its overall performance?

###### 5.1. Experimental Setup

Goal & Task. Given a query and a list of candidate documents retrieved by an initial, potentially weaker retriever, the goal is to identify the most relevant document(s) from within that list by processing the entire list in the LLM’s context.

Datasets & Formatting. For assessing zero-shot generalization, we use 11 diverse datasets from the BEIR benchmark (Thakur et al., 2021) replicating Table 1 in Reddy et al. (2024). In this setting, the task is to rerank the top-100 documents provided by Contriever (Izacard et al., 2021) model. For indomain analysis, we use two standard passage retrieval benchmarks: MSMarco Passage Ranking (Bajaj et al., 2018) and Natural Questions (NQ) (Kwiatkowski et al., 2019). During training, we construct candidate lists for each query by retrieving an initial set of 30 passages using a pre-trained sentence transformer model with teacher-forcing (i.e. always adding ground-truth documents). This list is then formatted into the prompt structure shown in Figure 3. During in-domain evaluation, we construct lists of varying sizes (𝑁 = 10 to 500) to test scalability. More details can be found in Appendix B.

Evaluation. We evaluate model performance on two primary aspects: effectiveness and efficiency. For BEIR, effectiveness is measured using nDCG@10. For in-domain experiments on MSMarco and NQ, we report Precision@1 and Mean Reciprocal Rank @ 10 (MRR@10). Efficiency is quantified by Inference Latency, the end-to-end wall-clock time per query.

- Table 1 | nDCG@10 on BEIR benchmark, all re-ranker rank top-100 documents retrieved from Contriever retrieval model. Bold indicates the best numbers.

Hotpot QA

DBPedia FEVER FiQA

Scidocs

Scifact

MS Marco

TrecCOVID

NFCorpus NQ

Train Data Avg.

ClimateFEVER

Reranker

None (Contriever) MS Marco 45.9 23.7 41.3 75.8 32.9 63.8 40.7 32.8 49.8 16.5 67.7 59.6 Cross-Encoder MS Marco 50.7 25.5 47.0 81.9 35.6 71.8 47.0 34.5 57.6 17.0 69.1 71.0 Rank Vicuna GPT 3.5 50.7 28.2 50.0 81.0 35.9 73.5 36.7 33.1 58.6 18.4 70.5 71.3 Rank Zephyr GPT 3.5+4 53.7 25.6 50.0 80.1 42.2 71.6 42.7 37.7 65.6 20.5 76.7 78.4 FIRST GPT-4 54.3 26.7 50.9 81.7 42.2 74.2 44.4 37.4 66.4 20.4 74.6 78.8

BlockRank Mistral MS Marco 54.8 26.8 49.7 87.3 44.9 75.5 48.6 36.6 62.4 18.7 76.5 76.2

Baselines. We compare BlockRank against a comprehensive set of baselines tailored to each experimental setting. For the BEIR generalization benchmark, we compare against contemporary listwise rerankers, including a strong cross-encoder, RankVicuna (Pradeep et al., 2023a), RankZephyr (Pradeep et al., 2023b), and the recent state-of-the-art model, FIRST (Reddy et al., 2024). For in-domain analysis, our primary comparison is with Full Fine-tuning (Full-FT) (full causal attention with only NTP loss) of the same base model and the same training data. We also include results from zero-shot LLMs (Mistral-7B-Instruct, Gemini-2.0-flash). For broader context, we include Traditional Retrieval Models such as the lexical baseline BM25, the dense retriever GTR (Ni et al., 2021), ColBERTv2 (Santhanam et al., 2022), and best performing Sentence Transformer Encoders specific to each dataset (msmarco-distilbert-dot-v5 for MSMarco and all-MiniLM-L12-v2 for NQ). Furthermore, we consider pairwise cross-encoder baselines like monoBERT (Nogueira and Cho, 2020) and improved versions of monoT5 (Nogueira et al., 2020).

Implementation Details. BlockRank and the Full-FT baseline utilize Mistral-7B-v0.3 as the base model. For fine-tuning both models, we employ the Adafactor optimizer (Shazeer and Stern, 2018) with a learning rate of 3 × 10−7 and a global batch size of 32 (accumulated across replicas). Each model is trained for 1 epoch with a linear warmup followed by cosine decay. For BlockRank, the auxiliary loss weight 𝜆 is set to 0.1, and 𝜏 is set to 0.05. Unless stated otherwise, BlockRank results employ the proposed attention-based inference. Decoding based experiments with BlockRank, LLM baselines (Full-FT Mistral and Zero-Shot LLMs) utilize greedy decoding to generate the relevant document identifier(s); to get multiple predictions (for MRR@10 evaluation) we use constrained beam decoding with beam-size set to 10, where only valid outputs are generated. All LLM fine-tuning and inference experiments were conducted using JAX on Google Cloud TPUs (specifically, 8 chip v6e configuration), and reported efficiency metrics correspond to this setup as well.

###### 5.2. Main Performance Comparison

Generalization to Diverse Tasks (RQ1) The results in Table 1 show that MSMarco-trained BlockRank Mistral (54.8) outperforms FIRST (54.3), RankZephyr (53.7), and RankVicuna (50.7), demonstrating strong out-of-distribution generalization. Importantly, BlockRank achieves strong results with the significant efficiency gains (Figure 4), presenting a compelling combination of effectiveness and scalability. Furthermore, it gets the strong performance by processing the entire list of 100 candidate documents in a single forward pass instead of multiple sliding-window forward passes over the candidate set – which is required for other listwise ranking models. These results also indicate that BlockRank is not sensitive to the first-stage retriever, as it effectively ranks candidates from Contriever despite its training data being constructed with a different retrieval model.

- Table 2 | Comparison on MSMarco and NQ datasets in controlled settings. Encoder methods are evaluated on the full corpus while the rest of the baselines are evaluated on a shortlist. Best results are highlighted in Bold.

NQ MSMarco

Category Method Model Size

Precision@1 Precision@1 MRR@10 Sparse Retrieval BM25 - 29.7 - 18.4

Sentence-transformer (Reimers and Gurevych, 2019) 66M 58.8 24.8 37.2 GTR-XXL (Ni et al., 2021) 4.8B - - 38.8 ColBERTv2 (Santhanam et al., 2022) 110M - - 39.7

Dual-Encoder

monoBERT (Nogueira and Cho, 2020) 110M - - 38.2 monoT5-XL (Nogueira et al., 2020) 3B - - 41.2

Cross-Encoder

Mistral-7B-v0.3-it (Jiang et al., 2023) 7B 43.5 13.1 Gemini-2.0-flash (Team et al., 2023) - 65.1 16.9 -

Zero-Shot LLM

Full-FT Mistral 7B 75.5 28.7 38.3 BlockRank Mistral (Ours) 7B 76.2 29.1 42.0

Fine-tuned LLM

In-Domain Performance (RQ1) Table 2 summarizes the quality comparisons on the NQ and MSMarco in a controlled environment where both BlockRank and Full-FT Mistral are trained on the same training data and evaluated on in-domain data, for broader comparison we also provide results for additional baselines trained on the same data. Our proposed BlockRank consistently outperforms its direct counterpart, Full-FT Mistral (7B).

Scalability (RQ2). Figure 4 underscores BlockRank’s substantial inference efficiency advantage over the Full-FT baseline as the number of in-context documents (𝑁) increases. BlockRank model consistently exhibits lower latency;

- at 𝑁 = 100, it is approximately 4.7× faster. More critically, its latency scales linearly with 𝑁, reaching 1.15s at 𝑁 = 500. Furthermore, BlockRank model maintains its P@1 (peaking around 29.2% for 𝑁 = 200 and remaining at 28.7% for 𝑁 = 500), whereas Full-FT’s P@1 sharply degrades beyond 𝑁 = 100 (dropping to ≈ 26.7% at 𝑁 = 500).

Figure 4 | P@1 and Latency (annotated) of BlockRank vs Full-FT Mistral, scaling 𝑁 on MSMarco.

30.0

29.5

448 ms

112 ms

52 ms

226 ms

1.15 s

32 ms

29.0

28.5

304 ms

1.07 s

P@1

118 ms

28.0

59 ms

27.5

BlockRank

27.0

Full-FT

26.5

101 102

Number of In-context Documents (N)

- 5.3. Ablation Studies Table 3 | Impact of training loss on Attention-based (Attn) and Decoding (Decode) Inference.

To understand contribution of components of BlockRank (RQ3), we perform several ablation experiments, primarily on the MSMarco dataset with 𝑁 = 50. More ablation is provided in Section D of Appendix.

Precision@1 Decode Attn

Training Configuration

Full-FT 28.7 27.6 Full-FT (w/ aux) 28.7 28.1

BlockRank (w/o ntp) 15.8 28.6 BlockRank (w/o aux) 28.4 27.8 BlockRank (full) 28.7 29.1

Impact of Training Loss Table 3 ablates the contributions of L𝑁𝑇𝑃 and L𝑎𝑢𝑥 to P@1, evaluated with both auto-regressive and attention-

based inference. Introducing L𝑎𝑢𝑥 consistently enhances performance for attention-based inference, and for BlockRank (which incorporates structured attention), it increases from 27.8 to 29.1. As expected, the L𝑁𝑇𝑃 objective is crucial for generative decoding performance, as seen by the sharp drop in Decode Prec@1 for ‘BlockRank (w/o ntp)’ to 15.8. Notably, the full BlockRank configuration achieves the highest Attn Prec@1 (29.1), demonstrating that L𝑎𝑢𝑥 effectively optimizes attention scores for direct retrieval, making attention-based inference the preferred mode for our method.

Table 4 | Ablation: Inference Method Effectiveness & Latency (MSMarco, N=50).

Impact of Inference Method Table 4 ablates the inference, comparing decoding against our attention-based approach on P@1 and MRR@10; for Full-FT, Decode MRR@10 uses a beam size of 10. The results show that while auto-regressive decoding yields comparable P@1 for both Full-FT (28.7) and BlockRank (28.7) models, it is significantly less effective at producing a strong ranked list for MRR@10. In contrast, BlockRank with attention-based inference performs best, achieving a notably better MRR@10 (42.0). BlockRank’s attention-based inference, optimized via its auxiliary loss, is more calibrated at assigning relevance across multiple predictions.

Model Inference P@1 MRR@10 Method

Full-FT Decode 28.7 38.4 Full-FT Attn 27.6 38.8

BlockRank Decode 28.7 40.0 BlockRank Attn 29.1 42.0

### 6. Conclusion

This work addresses the efficiency challenge in In-Context Retrieval (ICR) by analyzing LLM attention, identifying structured sparsity and query-token retrieval signals. We introduced BlockRank, a method that enforces this task-specific sparsity for linear complexity and uses a contrastive auxiliary loss to directly optimize these internal attention signals for relevance. Experiments on MSMarco and NQ show BlockRank (Mistral-7B) matches or surpasses standard fine-tuning effectiveness while being significantly more efficient at inference and training. This offers a scalable and effective approach for LLM-based ICR. However, we acknowledge our current findings are primarily demonstrated on a specific model architecture, and the robustness of the learned attention signals for direct inference across highly diverse tasks needs more investigation.

### References

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,

- S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Majumder, A. McNamara, B. Mitra,

- T. Nguyen, M. Rosenberg, X. Song, A. Stoica, S. Tiwary, and T. Wang. Ms marco: A human generated machine reading comprehension dataset, 2018. URL https://arxiv.org/abs/1611.09268.

- I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

- S. Chen, B. J. Gutiérrez, and Y. Su. Attention in large language models yields efficient zero-shot re-rankers, 2025. URL https://arxiv.org/abs/2410.02642.
- T. Formal, B. Piwowarski, and S. Clinchant. Splade: Sparse lexical and expansion model for first stage

ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2288–2292, 2021.

O. Goldman, A. Jacovi, A. Slobodkin, A. Maimon, I. Dagan, and R. Tsarfaty. Is it really long context if all you need is retrieval? towards genuinely difficult long context nlp, 2024. URL https: //arxiv.org/abs/2407.00402.

G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.

- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

- V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. T. Yih. Dense passage retrieval for open-domain question answering. In 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, pages 6769–6781. Association for Computational Linguistics (ACL), 2020.

T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026/.

- J. Lee, A. Chen, Z. Dai, D. Dua, D. S. Sachan, M. Boratko, Y. Luan, S. M. Arnold, V. Perot, S. Dalmia, et al. Can long-context language models subsume retrieval, rag, sql, and more? arXiv preprint arXiv:2406.13121, 2024.

X. Ma, X. Zhang, R. Pradeep, and J. Lin. Zero-shot listwise document reranking with a large language

#### model, 2023. URL https://arxiv.org/abs/2305.02156.

- J. Ni, C. Qu, J. Lu, Z. Dai, G. H. Ábrego, J. Ma, V. Y. Zhao, Y. Luan, K. B. Hall, M.-W. Chang, and Y. Yang. Large dual encoders are generalizable retrievers, 2021. URL https://arxiv.org/abs/ 2112.07899.

R. Nogueira and K. Cho. Passage re-ranking with bert, 2020. URL https://arxiv.org/abs/1901. 04085.

R. Nogueira, Z. Jiang, and J. Lin. Document ranking with a pretrained sequence-to-sequence model,

#### 2020. URL https://arxiv.org/abs/2003.06713.

R. Pradeep, S. Sharifymoghaddam, and J. Lin. Rankvicuna: Zero-shot listwise document reranking

with open-source large language models, 2023a. URL https://arxiv.org/abs/2309.15088. R. Pradeep, S. Sharifymoghaddam, and J. Lin. Rankzephyr: Effective and robust zero-shot listwise

#### reranking is a breeze!, 2023b. URL https://arxiv.org/abs/2312.02724.

- R. G. Reddy, J. Doo, Y. Xu, M. A. Sultan, D. Swain, A. Sil, and H. Ji. First: Faster improved listwise reranking with single token decoding. arXiv preprint arXiv:2406.15657, 2024.

N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks, 2019.

URL https://arxiv.org/abs/1908.10084.

- S. Robertson, H. Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.

#### K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction, 2022. URL https://arxiv.org/abs/2112.01488.

N. Shazeer and M. Stern. Adafactor: Adaptive learning rates with sublinear memory cost. Proceedings of the 35th International Conference on Machine Learning (ICML), 80:4596–4604, 2018.

W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren. Is chatgpt good at search? investigating large language models as re-ranking agents. arXiv preprint arXiv:2304.09542, 2023.

- M. S. Tamber, R. Pradeep, and J. Lin. Scaling down, litting up: Efficient zero-shot listwise reranking with seq2seq encoder-decoder models, 2023. URL https://arxiv.org/abs/2312.16098.

R. Tang, X. Zhang, X. Ma, J. Lin, and F. Ture. Found in the middle: Permutation self-consistency improves listwise ranking in large language models. arXiv preprint arXiv:2310.07712, 2023.

G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021.

J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.

- W. Wu, Y. Wang, G. Xiao, H. Peng, and Y. Fu. Retrieval head mechanistically explains long-context factuality, 2024. URL https://arxiv.org/abs/2404.15574.

- L. Xiong, C. Xiong, Y. Li, K.-F. Tang, J. Liu, P. Bennett, J. Ahmed, and A. Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval, 2020. URL https://arxiv.org/ abs/2007.00808.
- M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

- X. Zhang, S. Hofstätter, P. Lewis, R. Tang, and J. Lin. Rank-without-gpt: Building gpt-independent listwise rerankers on open-source large language models, 2023. URL https://arxiv.org/abs/ 2312.02969.

### A. Societal Impact

The BlockRank methodology, by enhancing the efficiency and scalability of In-context Retrieval (ICR) in Large Language Models (LLMs), makes advanced semantic retrieval more computationally tractable and can democratize access to powerful information discovery tools. This could accelerate research, improve educational outcomes by providing more relevant information quickly, and empower individuals and organizations with better decision-making capabilities. Furthermore, the increased efficiency directly translates to reduced energy consumption for retrieval-intensive LLM applications, contributing to more environmentally sustainable AI development and deployment. By enabling effective ICR on potentially smaller or more optimized models, BlockRank could also broaden the reach of these technologies in resource-constrained environments.

However, like many advancements in AI, more efficient information retrieval also presents challenges. The underlying LLMs can inherit and potentially amplify societal biases present in their training data. Therefore, continued research in this area should be accompanied by a strong emphasis on transparency, and the development of robust mechanisms to identify and mitigate the spread of harmful or misleading content.

### B. Dataset and Hyperparameter Details

This section provides a detailed description of the dataset and hyperparameters used in this study to ensure reproducibility.

- B.1. Datasets We use two standard passage retrieval benchmarks:

- • MSMarco Passage Ranking (Bajaj et al., 2018): We use MSMarco v1 passage retrieval dataset, it has total 8.8𝑀 passages, ∼ 500𝐾 training queries and 6980 validation queries. We directly utilize the hard negatives collection1 from huggingface for training. During test we retrieve the top-𝑁 passages using msmarco-distilbert-dot-v5 sentence-transformer.
- • Natural Questions (NQ320K) (Kwiatkowski et al., 2019): We use NQ320K passage retrieval dataset which has ∼ 320𝐾 passages in the corpus, ∼ 300𝐾 training queries and 7830 validation queries. For NQ, we collect hard negatives using all-MiniLM-L12-v2 sentence-transformer model for training. We use the same model during inference as well to retrieve top-𝑁 passages.

- B.2. Fine-tuning Details (BlockRank and Full-FT) The following fine-tuning settings were used for both BlockRank and Full-FT Mistral-7B:

- • Optimizer: Adafactor (Shazeer and Stern, 2018) with 𝛽1 = 0.9.
- • Learning Rate: 3 × 10−7.
- • Learning Rate Schedule: Linear warmup for 50 steps followed by a cosine decay.
- • Batch Size: A global batch size of 32, accumulated across replicas.
- • Number of Epochs: 1 epoch.
- • Weight Decay: No weight decay.
- • Gradient Clipping: gradient norm clipped to 1.0.
- • Loss for Full-FT: Standard Next Token Prediction (NTP) cross-entropy loss, calculated on the answer tokens (i.e., the ID of the relevant document).

1https://huggingface.co/datasets/sentence-transformers/msmarco-msmarco-distilbert-base-v3

###### B.3. BlockRank Specific Hyperparameters

In addition to the general fine-tuning settings, the following hyperparameters are specific to the BlockRank method:

- • Auxiliary Loss Weight (𝜆): The hyperparameter balancing the NTP loss and the auxiliary attention loss (L𝑎𝑢𝑥) was set to 𝜆 = 0.1 this ensures that both loss have the same scale.
- • InfoNCE Temperature (𝜏): The temperature parameter for the InfoNCE loss (L𝑎𝑢𝑥) was set to 𝜏 = 0.05.
- • Signal-Carrying Query Tokens (𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙): Based on our prompt template and empirical analysis (Section D), the set of tokens for signal-carrying query tokens was 𝑇𝑞,𝑠𝑖𝑔𝑛𝑎𝑙 = [":", "[‘"].
- • Middle Layer for Auxiliary Loss (𝑙∗): The auxiliary loss L𝑎𝑢𝑥 was applied at a specific middle layer 𝑙∗ = 20, determined empirically as described in Section D.3.
- • Chunk Length (𝐿𝑐ℎ𝑢𝑛𝑘): The fixed length for chunks used in the structured attention mechanism. We set 𝐿𝑐ℎ𝑢𝑛𝑘 = 160 for MSMarco and 𝐿𝑐ℎ𝑢𝑛𝑘 = 384 for NQ, this ensures that ∼ 95% of the passages get full represented in 𝐿𝑐ℎ𝑢𝑛𝑘 sequence length.

C. Attention Complexity Analysis

This section provides a analysis of the computational complexity of the structured attention mechanism within a single layer of the BlockRank model architecture. Our aim is to clearly illustrate the scalability benefits of BlockRank, particularly its linear scaling with respect to the number of candidate documents, 𝑁. We define 𝐿𝑐ℎ𝑢𝑛𝑘 as the fixed characteristic length (number of tokens) for segments after processing, and 𝑑 as the hidden dimension of the model. For this analysis, we assume that the instruction segment, each of the 𝑁 document segments, and the query segment have effective lengths 𝐿𝐼𝑛𝑠𝑡 = 𝐿𝑐ℎ𝑢𝑛𝑘, 𝐿𝑑𝑜𝑐 = 𝐿𝑐ℎ𝑢𝑛𝑘, and 𝐿𝑞 = 𝐿𝑐ℎ𝑢𝑛𝑘 respectively, when their attention computations are considered. This section focuses exclusively on the attention component’s complexity, as this is where BlockRank introduces its primary architectural modification for efficiency.

The BlockRank model implements a structured sparse attention mechanism, as detailed in Section

###### 4.1 of the main paper, where different parts of the input prompt adhere to distinct attention patterns. The instruction segment, with its effective length of 𝐿𝑐ℎ𝑢𝑛𝑘, performs causal self-attention, leading

to a complexity of 𝐶𝑎𝑡𝑡𝑛,𝐼𝑛𝑠𝑡 = O(𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑). For the 𝑁 document segments, each also of effective length 𝐿𝑐ℎ𝑢𝑛𝑘, tokens attend both within their own segment and to tokens within the instruction segment. This means the effective context length for a token in any given document segment is 𝐿𝑐ℎ𝑢𝑛𝑘 + 𝐿𝑐ℎ𝑢𝑛𝑘 = 2𝐿𝑐ℎ𝑢𝑛𝑘. Consequently, the attention complexity for a single document segment is O(𝐿𝑐ℎ𝑢𝑛𝑘 · 2𝐿𝑐ℎ𝑢𝑛𝑘 · 𝑑) = O(2𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑). Summing across all 𝑁 document segments, their total attention complexity is 𝐶𝑎𝑡𝑡𝑛,𝐷𝑜𝑐 = 𝑁 · O(2𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑).

The query segment, also with an effective length of 𝐿𝑐ℎ𝑢𝑛𝑘, has the full attention scope. It attends to its own tokens, tokens from the instruction segment, and tokens from all 𝑁 document segments. The total context length for these query tokens becomes 𝐿𝑐ℎ𝑢𝑛𝑘 + 𝐿𝑐ℎ𝑢𝑛𝑘 + (𝑁 · 𝐿𝑐ℎ𝑢𝑛𝑘) = (𝑁 + 2)𝐿𝑐ℎ𝑢𝑛𝑘. The attention complexity for the query segment is therefore 𝐶𝑎𝑡𝑡𝑛,𝑄𝑢𝑒𝑟𝑦 = O(𝐿𝑐ℎ𝑢𝑛𝑘 · (𝑁 + 2)𝐿𝑐ℎ𝑢𝑛𝑘 · 𝑑) = O((𝑁 + 2)𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑).

Summing the complexities of these components gives the total attention complexity per layer for the BlockRank model, 𝐶𝑎𝑡𝑡𝑛,BlockRank:

𝐶𝑎𝑡𝑡𝑛,BlockRank = 𝐶𝑎𝑡𝑡𝑛,𝐼𝑛𝑠𝑡 + 𝐶𝑎𝑡𝑡𝑛,𝐷𝑜𝑐 + 𝐶𝑎𝑡𝑡𝑛,𝑄𝑢𝑒𝑟𝑦 𝐶𝑎𝑡𝑡𝑛,BlockRank = O(𝐿2𝑐ℎ𝑢𝑛𝑘𝑑) + 𝑁 · O(2𝐿2𝑐ℎ𝑢𝑛𝑘𝑑) + O((𝑁 + 2)𝐿2𝑐ℎ𝑢𝑛𝑘𝑑)

This simplifies to O(3𝐿2𝑐ℎ𝑢𝑛𝑘𝑑 + 3𝑁𝐿2𝑐ℎ𝑢𝑛𝑘𝑑), which is O((𝑁 + 1)𝐿2𝑐ℎ𝑢𝑛𝑘𝑑). The dominant term thus yields a total attention complexity of 𝐶𝑎𝑡𝑡𝑛,BlockRank = O(𝑁 · 𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑). This result clearly shows that the attention complexity in the BlockRank architecture scales linearly with 𝑁, the number of documents.

In contrast, a standard Transformer model processing a sequence of comparable total length 𝑆 ≈ (𝑁 + 2)𝐿𝑐ℎ𝑢𝑛𝑘 would exhibit an attention complexity of 𝐶𝑎𝑡𝑡𝑛,𝑆𝑡𝑑 = O(𝑆2 · 𝑑). For large 𝑁, this is approximately O(((𝑁 + 2)𝐿𝑐ℎ𝑢𝑛𝑘)2 · 𝑑) = O(𝑁2 · 𝐿2𝑐ℎ𝑢𝑛𝑘 · 𝑑), which is quadratic with respect to 𝑁.

### D. Additional Results

###### D.1. Calibration Problem in Beam Decoding with Full-FT Model

Table 5 | Entropy of predicted document ID digits (𝑖𝑑0 and 𝑖𝑑1) over 10 beam-decoded predictions for the Full-FT model versus random predictions. Lower entropy indicates less diversity in the generated digits across the prediction list.

To analyze the behavior of the standard finetuned (Full-FT) model when generating multiple distinct predictions via beam decoding, we conducted an entropy analysis on the individual tokens of the predicted document identifiers. This experiment was designed to assess the diversity of predictions for structured identifiers (two-digit IDs from 0-99, given 𝑁 = 100 candidate documents). For each query in the test set, we generated 10 unique document ID predictions from the Full-FT model. We then computed the entropy of the distribution of the first digit (𝑖𝑑0) and the second digit (𝑖𝑑1) across these 10 predictions. This was compared against the entropy derived from 10 randomly drawn unique two-digit IDs. Because the candidate list is randomly shuffled and the ID assigned to each document is completely independent from its content, a lower entropy would indicate a undesirable concentration of predicted digits, suggesting a lack of diversity in the generated list beyond the top few candidates.

Prediction Model Entropy 𝑖𝑑0 Entropy 𝑖𝑑1 Full-FT 2.28 ± 0.43 2.19 ± 0.46 BlockRank 2.54 ± 0.24 2.67 ± 0.24 Random 2.55 ± 0.25 2.66 ± 0.24

The results, summarized in Table 5, show that the Full-FT model exhibits lower average entropy for both 𝑖𝑑0 (2.28 ± 0.43) and 𝑖𝑑1 (2.19 ± 0.46) compared to the random baseline (2.55 ± 0.25 for 𝑖𝑑0 and 2.66 ± 0.24 for 𝑖𝑑1). This decreased entropy indicates that the sequence of document identifiers generated by the Full-FT model via beam decoding tends to be less diverse in its constituent digits than random chance would suggest. To give an example, we observe that let’s say the model predicts 73 as it’s top prediction with high confidence, then there is a high likelihood that it will predict other IDs either starting with 7 or ending with 3. Such concentration implies that while the model may identify a strong top candidate, its ability to produce a well-calibrated and varied set of subsequent predictions is limited, due to the nature of auto-regressive log-probability distributions. This observation supports the main paper’s discussion (Section 5.3, Table 4) on the sub-optimality of beam decoding for generating ranked lists for ICR.

###### D.2. Analysis of Retrieval Signals in Attention Patterns of Full-FT Mistral

To substantiate the claims made in Section 3 of the main paper regarding the presence of retrieval signals within the internal attention patterns of a standard fine-tuned (Full-FT) language model, we conducted a series of analytical experiments. These experiments, detailed below, confirm the characteristics of such signals using attention-based inference on the MSMarco dev dataset.

First, we investigate the specific carriers of the retrieval signals. Figure 5 presents the P@1 performance of attention-based inference when attention scores are extracted from different query tokens within the prompt. This analysis reveals that certain query tokens, particularly those located towards the end of the query or specific delimiter tokens such as “:” and terminal prompt markers, serve as strong "signal carriers," yielding significantly higher P@1 when their attention patterns are used to predict the relevant document.

[Figure 5]

Figure 5 | Performance of Full-FT model’s attention-based inference vs the query token for which attention scores are extracted from.

[Figure 6]

- Figure 6 | Performance of Full-FT model’s attention-based inference as a function of the Transformer Layer Index from which attention scores are extracted (MSMarco).

Complementing this, Figure 6 evaluates both P@1 and Mean Reciprocal Rank @10 for attentionbased inference as a function of the Transformer layer index from which attention scores are derived. This experiment confirms that the retrieval signal is most prevalent in the middle layers of the Full-FT model, with performance declining in earlier and later layers. Collectively, these empirical findings demonstrate that standard LLMs fine-tuned for In-context Retrieval exhibits latent retrieval signals within their attention mechanisms. These signals are characterized by their preferential emergence in middle layers and their association with specific query tokens.

D.3. Layerwise Emergence of Retrieval Signals and Choice of 𝑙∗

[Figure 7]

Figure 7 | Layerwise Attention Precision@1 on a held-out subset of MSMarco training data vs training steps for Full-FT model

- Figure 7 illustrates the evolution of layerwise Precision@1 derived from attention scores on a held-out subset of MSMarco training data as the Full-FT model undergoes training. It is observed that effective retrieval signals, as measured by the attention-P@1 metric, do not develop uniformly across all layers. Instead, they emerge more prominently and strengthen considerably in the middle layers of the transformer (layers 12 through 24) as training progresses, while shallower and deeper layers exhibit comparatively weaker signal strength. Based on this we set the 𝑙∗ = 20 for all of our BlockRank experiments. Although, we find that the choice of 𝑙∗ in BlockRank is not very sensitive to this specific layer, any reasonable middle layer gives similar performance.

###### D.4. Impact of Including Query in Prompt Prefix

Table 6 | Ablation on Including Query in Prompt Prefix. Comparison of Precision@1 on MSMarco (N=100) for Full-FT and BlockRank models.

We investigated whether providing the query context upfront, in addition to its standard position at the end of the prompt, impacts retrieval performance. Table 6 compares the Precision@1 results on MSMarco for the Full-FT baseline and our BlockRank model using the standard format (query only at end; denoted by ✗ in the table) versus the query-prefix format (query at beginning and end; denoted by ✓). Including the query redundantly in the prefix (✓) improved performance over the standard format (✗) for both models. The Full-FT model’s Prec@1 increased from 27.2 to 28.7 (+1.5), while our BlockRank model saw a more substantial increase from 24.2 to 28.1 (+3.9). This suggests that priming the model with the query context before it processes the candidate documents is beneficial, perhaps allowing attention mechanisms, particularly the specialized ones in BlockRank, to focus more effectively on query-relevant information throughout the sequence. Given this clear advantage, we utilize the prompt format that includes the query in the prefix for all other reported experiments.

Model Query in Prefix Prec@1

Full-FT-Mistral ✗ 27.2 Full-FT-Mistral ✓ 28.7

BlockRank-Mistral ✗ 24.2 BlockRank-Mistral ✓ 29.1

###### D.5. Cross-dataset Generalization

Table 7 | Cross-dataset generalization performance of BlockRank Mistral models. P@1 scores are reported on the NQ and MSMarco test sets for models with no training (zero-shot Mistral-7B-instruct), fine-tuned on NQ, and fine-tuned on MSMarco.

To assess the generalization of the BlockRank models, we evaluated BlockRank Mistral models trained on one dataset and tested on another, unseen dataset. Specifically, models were finetuned separately on the MSMarco and Natural Questions (NQ) training sets, and their Precision@1 (P@1) performance was subsequently measured on the test sets of both NQ and MSMarco. For reference, we also include the performance of a zero-shot Mistral-7B-instruct model (denoted as No Training). The results of this cross-dataset evaluation are presented in Table 7. As expected, BlockRank models achieve their best performance when evaluated on the in-domain test set. When evaluated on out-of-domain datasets, the performance, shows positive transfer above the No Training baseline but is considerably lower than in-domain scores.

##### Training Data NQ P@1 MSMarco P@1

No Training 43.5 13.1 NQ 76.2 18.2 MSMarco 62.0 29.1

