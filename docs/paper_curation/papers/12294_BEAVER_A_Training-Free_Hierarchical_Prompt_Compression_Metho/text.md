## BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection

Zhengpei Hu1*, Kai Li2,∗, Dapeng Fu3, Chang Zeng, Yue Li1, Yuanhao Tang1, Jianqiang Huang1† 1School of Computer Technology and Application, Qinghai University 2Tsinghua University 3Ant Group Security and Intelligence Laboratory (SIL)

# arXiv:2603.19635v1[cs.CL]20Mar2026

#### Abstract

The exponential expansion of context windows in LLMs has unlocked capabilities for longdocument understanding but introduced severe bottlenecks in inference latency and information utilization. Existing compression methods often suffer from high training costs or semantic fragmentation due to aggressive token pruning. In this paper, we propose BEAVER, a novel training-free framework that shifts compression from linear token removal to structure-aware hierarchical selection. BEAVER maximizes hardware parallelism by mapping variable-length contexts into dense page-level tensors via dualpath pooling, and preserves discourse integrity through a hybrid planner combining semantic and lexical dual-branch selection with sentence smoothing. Extensive evaluations on four longcontext benchmarks demonstrate that BEAVER achieves comparable performance to state-ofthe-art (SOTA) methods like LongLLMLingua. Notably, on the RULER benchmark, BEAVER maintains high fidelity in multi-needle retrieval where baselines deteriorate. Regarding efficiency, BEAVER reduces latency by 26.4× on 128k contexts, offering a scalable solution for high-throughput applications. Our code is available at https://cslikai.cn/BEAVER/.

#### 1 Introduction

In recent years, the context window size of LLMs has expanded exponentially, evolving from the early 32k token configuration to the million-token scale supported by models such as Claude 4.5 (Anthropic, 2025) and Gemini 3.0 (DeepMind, 2025). This advancement enables powerful capabilities for codebase analysis (Li et al., 2024, 2025a) and global understanding across multiple long documents (Ma et al., 2024; Deng et al., 2025), while simultaneously revealing several critical bottlenecks that hinder practical deployment.

* Equal contribution. † Corresponding author.

The first challenge is the "computation wall" in inference phase (Dao et al., 2022; Zhao et al., 2025). While quantization (Lin et al., 2024) and system-level optimizations (Kwon et al., 2023) reduce memory pressure, the O(L2) complexity of self-attention still causes prefill latency to surge with context length, leading to excessive time to first token and tail latency. The second challenge is the "diminishing returns" in information utilization (Sinha et al., 2025). Extending context windows does not yield proportional gains and often triggers the "Lost in the Middle" effect (Liu et al., 2024), where models overlook key information. These challenges underscore that scaling length alone cannot ensure robustness, necessitating a new paradigm that balances efficiency and accuracy.

To address these challenges, prompt compression (Li et al., 2025b) offers an efficient solution by pruning redundancy while preserving semantics. Existing approaches fall into two paradigms (1): (i) Unsupervised statistical methods, such as SelectiveContext (Li et al., 2023a) and the LongLLMLingua (Jiang et al., 2023), which filter tokens based on perplexity (PPL) or self-information from off-theshelf models; and (ii) Supervised and specialized learning methods, like CPC (Liskavets et al., 2025), LLMLingua (Jiang et al., 2023) and LLMLingua-2 (Pan et al., 2024), which treat compression as a classification or ranking task using trained encoders to achieve higher precision.

However, two limitations hinder practical deployment. First, reliance on training incurs significant overhead and limits cross-model generalization, undermining the goal of lightweight inference. Second, unstructured token pruning disrupts semantic and syntactic coherence, resulting in fragmented contexts that impede long-sequence modeling.

To address two these limitations, we propose BEAVER, a novel training-free framework that advances from linear token-wise pruning to structureaware hierarchical selection, as shown in Figure 2.

Context:Important Notice: Valid for VIP members. For regular customers, add 20% to the listed prices.... [Omitted 30k Tokens] ... the iPhone 17 Pro is listed at $899.The iPad Air is listed at $599.Additionally... [Omitted] ... Note: This promotion ended on Nov 12th. Query: What is the promotional price for the iPhone 17 Pro ?

[Figure 1]

###### (B) Supervised and Specialized Learning Methods

###### (C) Training-free and Zero-Overhead Methods

###### (A) Unsupervised Statistical Methods

Semantic

Datasets

Pre-trained Model

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Tuning

Target LLM Embedding

[Figure 6]

Fine-tuned Models

[Figure 7]

###### +

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

PPL Calculation

[Figure 12]

Human Aware

[Figure 13]

[Figure 14]

Aligned

Misaligned

Lexical

Important ... valid VIP ... regular ... add 20% ... Samsung Galaxy S24 Ultra ... stock ... Google

Important Notice: Valid for VIP members. For regular customers, add 20% to the listed prices.the iPhone 17 Pro is listed at $899.Note: This promotion ended on Nov 12th.

Samsung Galaxy S24 Ultra is out of stock. Google Pixel 8 is $699. iPhone 17 Pro is listed at $899.

Pixel 8 ... $699 ... iPhone 17 Pro ... $899 ... iPad Air ... $599 ... MacBook ... iCloud+ ... Nov 12th.

The iPad Air is listed at $599. MacBook purchases include iCloud+ subscription.

[Figure 15]

[Figure 16]

[Figure 17]

Fragmented & Noisy

Context-Lost & Misleading

Coherent & Faithful

Figure 1: Comparison of different prompt compression paradigms for long-context LLMs.

The framework comprises three key components: a Segmenter that converts sequences into 2D page tensors to optimize GPU efficiency and preserve discourse; a PageEncoder that employs dual-path pooling to capture hierarchical features; and a QueryPlanner that integrates a triple structural prior (anchors, flow, flash) to mimic human cognition and suppress semantic drift (Aytes et al., 2025). Finally, a smoothing mechanism restores selections into coherent, high-fidelity compressed text.

Extensive evaluations on four long-context benchmarks (LongBench (Bai et al., 2024), ZeroSCROLLS (Shaham et al., 2023), RULER (Hsieh

- et al., 2024), and L-Eval (An et al., 2024)) demonstrate that BEAVER achieves performance on par with or superior to SOTA methods like LongLLMLingua (Jiang et al., 2024) and LLMLingua-2 (Pan et al., 2024). Notably, BEAVER dominates the challenging RULER benchmark, nearly doubling the performance of existing baselines. In terms of efficiency, it yields a 26.4× speedup over LongLLMLingua at 128k context. Moreover, as a training-free framework, BEAVER proves robust across diverse model scales (0.6B–32B), positioning it as a scalable solution for high-throughput long-document understanding.

- 2 Related Work

With the expansion of context windows in LLMs, prompt compression has become a vital strategy to mitigate inference costs by pruning redundant information while preserving core semantics (Li

- et al., 2025b). Current research in hard prompt compression, which maintains discrete tokens for

interpretability and compatibility, can be categorized into two primary paradigms.

The first paradigm consists of unsupervised statistical methods that rely on the perplexity or generation probabilities of off-the-shelf models without specialized training. For instance, SelectiveContext (Li et al., 2023a) and the LongLLMLingua (Jiang et al., 2024) utilize self-information metrics to identify and prune redundant tokens or chunks. While efficient, these methods are often limited by their reliance on heuristic information-theoretic metrics. The second paradigm involves supervised and specialized learning methods, which optimize model weights specifically for compression objectives. LLMLingua and LLMLingua-2 (Jiang et al., 2023; Pan et al., 2024) reformulates compression as a token-level classification task via data distillation, while PCRL (Jung and Kim, 2024) and TACO-RL (Shandilya et al., 2025) employ reinforcement learning for task-aware filtering. Additionally, works like CPC (Liskavets et al., 2025), AdaComp (Zhang et al., 2024), and TCRA-LLM (Liu et al., 2023) leverage specialized semantic encoders or embeddings to perform relevance filtering at the sentence or paragraph level, thereby mitigating local semantic fragmentation.

However, these methods face a dichotomy: learning-based approaches suffer from high deployment costs and limited transferability, while finegrained token pruning disrupts syntactic coherence. Distinct from these methods, BEAVER introduces a training-free, structure-aware framework. By shifting from token-wise to segment-page selection, it effectively eliminates training dependencies

while preserving discourse integrity, addressing the fragmentation issues prevalent in prior methods.

#### 3 Methods

##### 3.1 Overall Pipeline

Given a long-context sequence C ∈ Z1×Lc and a query Q ∈ Z1×Lq, where Lc and Lq denote the number of tokens in the context and query respectively, our objective is to extract a compressed context C˜ ∈ Z1×Lp such that Lp ≪ Lc, while preserving the key information required to derive the correct answer A as much as possible. As shown in Figure 2, BEAVER consists of three components: Segmenter, PageEncoder, and QueryPlanner.

Specifically, BEAVER applies compression only to the context preceding the query Q. If Q is explicit, the prefix is compressed while Q is fully retained. If Q is implicit, it is defined as the final segment of the sequence, anchored to the nearest natural language boundary via a backward window to prevent truncation. The Segmenter then splits C into logical segments based on semantic delimiters (such as newlines or heading markers) and performs pagination to construct a structured two-dimensional page tensor P ∈ NN×M. On this basis, BEAVER introduces a dual-path pooling encoder (PageEncoder) equipped with a contextstatistics-adaptive weighting mechanism, which jointly encodes the paginated token tensor P to obtain page-level representations Pˆ ∈ RN×d, where d denotes the feature dimension. The QueryPlanner executes the core compression by computing a hybrid semantic-lexical interaction score between Pˆ and Q. It constructs a saliency mask by integrating structural priors, such as anchor pages and flow pages, with high-scoring pages from the score distribution. Subsequently, sentence-level smoothing is applied to the salient pages. Finally, the discrete mask is mapped back to a continuous subsequence C˜ for downstream LLM inference.

##### 3.2 Segmenter

The Segmenter is designed to efficiently map the variable-length sequence C into a 2D page matrix based on natural delimiters D (e.g., newlines). We first partition C into logical segments {c1,...,cK}. To construct the matrix, we employ a greedy pagination strategy with capacity M: multiple consecutive segments are packed into a single page if their cumulative length is within M, minimizing padding; longer segments are split across

consecutive pages. The output is a page index tensor P (padded with −1), which enables efficient standard matrix multiplication while preserving local semantic boundaries.

##### 3.3 PageEncoder

To enable QueryPlanner to efficiently perform coarse-grained selection, BEAVER uses a dualpath pooling encoder named PageEncoder. It captures both global semantics and salient local features through weighted average and max pooling while utilizing a context-adaptive weighting mechanism to mitigate semantic collapse caused by redundant tokens.

Specifically, given the LLM embedding E : N → Rd, we first map the token sequence to token features H ∈ RLc×d, where the feature of the ℓ-th token is hℓ = E(Cℓ). Using the page index tensor P generated by the Segmenter, we rearrange the token features into a page-level tensor X ∈ RN×M×d, where the feature at position (i,j) is defined as

hℓ, if Pi,j = ℓ ≥ 0, 0, if Pi,j = −1.

(1)

xi,j =

To down-weight frequent but uninformative tokens, we compute an in-context Inverse Term Frequency (ITF) score. Let tf(t) be the token frequency within the context, the weight is defined as:

Lc + Lq 1 + tf(t)

witf(t) = Norm log 1 +

, (2)

where Norm(·) denotes min-max normalization to [0,1]. The effective attention weight for position (i,j) is then ωi,j = Bi,j · witf(CPi,j), with binary mask Bi,j = I[Pi,j ≥ 0].

We then extract complementary representations via two pooling paths. The weighted average pooling aggregates global semantics:

M j=1 ωi,jxi,j M j=1 ωi,j + ε

, (3)

µi =

where ε ensures numerical stability. Simultaneously, the max pooling path captures salient local activations (e.g., rare keywords). We mask invalid positions with a large negative constant β to ensure numerical correctness:

mi = max

1≤j≤M

Bi,jxi,j + (1 − Bi,j) · β . (4)

| |
|---|

| |
|---|

Page Feature

Anchor Page Flow Page Raw Page Padding Token

Flash Page

PageEncoder

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Subject: Formal response regarding the incident on May 4th...

In-Context-ITF Calculator

2D Pages

Subject: Formal response regarding the incident on May 4th...... [Omitted 5k Tokens of legal details]

| | | | |
|---|---|---|---|

[...Omitted...]

Mean Pooling

| | | | |
|---|---|---|---|
| | | | |

...I never said she stole my car...

Segment

......I never said she stole my car... [Omitted] ......I can forgive you if you apologize sincerely.

Max Pooling

### ...

### ...

| | | | |
|---|---|---|---|
| | | | |
| | | | |

...I can forgive you if you apologize sincerely.

Original Context

Segmenter

Reprs

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Subject: Formal response regarding the incident on May 4th.I never said she stole my car.I can forgive you if you apologize sincerely.

Lexical Matcher

Did she steal the car and how to get forgiveness?

Sentence Smoother

Structure Aware

Semantic Matcher

Query

Hybrid Matcher

Page remains Heat Map

Compressed Context

| | |
|---|---|
|Heat| |

QueryPlanner

Figure 2: Overall pipeline of BEAVER. It comprises a Segmenter, a PageEncoder, and a QueryPlanner.

The final representation is a linear fusion

pˆi = γµi + (1 − γ)mi, (5)

where γ is fusion weights. All page representations are stacked into a matrix Pˆ ∈ RN×d.

Similarly, for short queries (length Lq < 4), we adopt a dense retrieval approach (Karpukhin et al., 2020) to obtain a unified semantic vector q ∈ Rd. For longer queries (Lq ≥ 4), we employ a late interaction strategy (Khattab and Zaharia, 2020), retaining fine-grained token representations to handle complex matching requirements.

##### 3.4 QueryPlanner

The QueryPlanner identifies salient pages from the encoded set Pˆ to construct a compressed context. To capture both semantic relevance and exact answer spans, it integrates dense retrieval and lexical overlap within a unified space, incorporated with structural priors.

Given the query representation Q = {qk}Kk=1 (where K = 1 for short queries) and the set of page vectors Pˆ = {pˆi}Ni=1, we compute a semantic score via a weighted cosine similarity. To handle multi-vector queries effectively, we define:

K

pˆi · qk ∥pˆi∥2 · ∥qk∥2

wqitf

, (6)

ssem(i) =

k

k=1

where wqitf

is the ITF score of qk. This mechanism naturally assigns higher importance to discriminative terms (e.g., entities) over common stop words.

k

Complementarily, the lexical branch highlights exact token overlaps. Let TQ be the query token set and Pi be the tokens in page i. We aggregate the importance of overlapping tokens:

slex(i) =

I[Cℓ ∈ TQ] · wCitf

. (7)

ℓ

ℓ∈Pi

Both scores are min-max normalized to [0,1] over all N pages. The final relevance score is a linear fusion smix(i) = λssem(i) + (1 − λ)slex(i), where λ is one hyperparameter that control the relative importance of semantic and lexical alignment. Tasks emphasizing semantic reasoning may assign a larger λ, while those containing many code snippets or identifiers may benefit from a smaller λ.

After computing the scores smix(i), QueryPlanner incorporates structural priors to enhance stability and discourse coherence. We first identify the query anchor pqry (the page containing the query start) to enforce causal constraints (i ≤ pqry). The selection set is composed of three distinct subsets: (1) Anchors: The initial kanc pages are always preserved to retain global metadata (e.g., title, introductory definitions). (2) Flow: To mimic human working memory (Johnson-Laird, 2010), we select a contiguous window size wflow immediately preceding the query, defined as indices i ∈ [max(1,pqry − wflow),pqry]. (3) Flash: From the remaining candidates (excluding Anchors and Flow), we select the top-kflash pages with the highest smix(i) to capture distant but critical evidence.

she stole my car.

if you apologize sincerely.

I never said I can forgive you

Figure 3: Illustration of sentence-level smoothing. To ensure syntactic completeness, the initially retrieved span (orange solid box) is extended outward to the nearest sentence boundaries (dashed boxes).

Finally, the selected page indices are mapped back to token spans [ar,br]. As illustrated in Figure 3, we apply sentence-level smoothing: start

- ar and end br indices are extended outward to the nearest sentence boundaries to preserve syntactic completeness. Overlapping spans are merged, and the result is concatenated to form the final input C˜.

#### 4 Experiments

##### 4.1 Experimental Setup

Benchmarks. To comprehensively evaluate the effectiveness and robustness of BEAVER, we conducted experiments on four diverse and challenging long-context benchmarks. We first adopted LongBench (Bai et al., 2024) and ZeroSCROLLS (Shaham et al., 2023), which cover multiple tasks including single-document and multi-document question answering, summarization, and few-shot learning, providing a holistic view of general longcontext capabilities. We then used the high-quality synthetic benchmark RULER (Hsieh et al., 2024) to test long-context understanding under varying context lengths from 16k to 128k, including tasks such as multi-needle retrieval and variable tracking, which precisely assess the model’s ability to retain fine-grained information. Finally, we further evaluated the generalization performance of BEAVER on out-of-domain tasks using L-Eval (An et al., 2024), which enforces strict token-length constraints. More detailed dataset descriptions and evaluation metrics are provided in Appendix A.

Baseline Models. To ensure a fair comparison, we evaluated BEAVER aagainst several prompt compression methods. The baselines are grouped into two categories: (1) Unsupervised statistical methods such as Selective-Context (Li et al., 2023a) and LongLLMLingua (Jiang et al., 2024), which rely on intrinsic metrics such as perplexity or selfinformation from off-the-shelf models to identify redundancy without specialized training. (2) Supervised and specialized learning methods, which utilize models explicitly optimized for compres-

sion tasks or leverage specialized semantic encoders such as LLMLingua (Jiang et al., 2023) and LLMLingua-2 (Pan et al., 2024). Additionally, we included embedding-based retrieval approaches, like SBERT (Reimers and Gurevych, 2019) and OpenAI Embeddings (OpenAI, 2024), within this paradigm as they employ trained encoders to perform semantic relevance filtering. Baseline configurations and implementation details are available in Appendix B.

Implementation Details. For the inference backend, we followed existing methods (Jiang et al., 2024; Pan et al., 2024) and uniformly adopted gpt-3.5-turbo-instruct1 as the large language model for all downstream task evaluations. The PageEncoder utilized embeddings from Qwen3-8B2. Regarding hyperparameters, we set the page size M = 64, fusion weight γ = 0.7, and scoring parameter λ = 0.7. Structural priors were fixed at kanc = wflow = 4, with the flash set size kflash dynamically adapted to fully utilize the remaining token budget. Our evaluations strictly adhered to benchmark-specific context budgets and employ official standard prompt templates to ensure fair comparison. All latency and throughput metrics were measured on NVIDIA A100 (80GB) GPUs.

4.2 Comparison with SOTA Methods 4.2.1 Performance Analysis

To demonstrate the advantages of BEAVER, we compared it with SOTA methods on four different benchmarks: LongBench (Bai et al., 2024), ZeroSCROLLS (Shaham et al., 2023), RULER (Hsieh et al., 2024), and L-Eval (An et al., 2024), under strict token budgets of 2,000 and 3,000 tokens.

As shown in Table 1, BEAVER consistently outperforms the latest learning-based LLMLingua2 (Pan et al., 2024) on LongBench, establishing a new SOTA of 40.7 on single-document QA. While LongLLMLingua (Jiang et al., 2024) excels on synthetic tasks, it suffers from high computational costs due to its coarse-to-fine re-ranking mechanism. In contrast, BEAVER is fully training-free and achieves comparable results on ZeroSCROLLS (average 32.0 vs. 32.7, see Appendix C for detailed analysis) with significantly reduced latency. Specifically, BEAVER achieves the lowest latency (3.0s) and highest speedup (5.2×), validating that

- 1https://platform.openai.com/docs/models/

gpt-3.5-turbo-instruct

- 2https://huggingface.co/Qwen/Qwen3-8B

LongBench ZeroSCROLLS

Methods

SingleDoc MultiDoc Summ. FewShot Synth. Code AVG Tokens 1/τ Lat. Spd. AVG Tokens 1/τ Lat. Spd.

- 2,000-token constraint

- (1) Unsupervised Statistical Methods Selective-Context 16.2 34.8 24.4 15.7 8.4 49.2 24.8 1,925 5x 47.1 0.3x 19.4 1,865 5x 47.5 0.3x

- LongLLMLingua 39.0 42.2 27.4 69.3 53.8 56.6 48.0 1,809 6x 6.1 2.6x 32.7 1,753 6x 5.2 2.3x

(2) Supervised and Specialized Learning Methods

SBERT 33.8 35.9 25.9 23.5 18.0 17.8 25.8 1,947 5x 4.8 3.4x 20.5 1,773 6x 4.1 3.0x OpenAI 34.3 36.3 24.7 32.4 26.3 24.8 29.8 1,991 5x 10.4 1.5x 20.6 1,784 5x 9.9 1.2x LLMLingua 22.4 32.1 24.5 61.2 10.4 56.8 34.6 1,950 5x 5.9 2.6x 27.2 1,862 5x 4.8 2.5x LLMLingua-2-small 29.5 32.0 24.5 64.8 22.3 56.2 38.2 1,891 5x 3.3 4.7x 33.3 1,862 5x 2.6 4.7x LLMLingua-2 29.8 33.1 25.3 66.4 21.3 58.9 39.1 1,954 5x 3.7 4.2x 33.4 1,898 5x 3.0 4.1x

(3) Training-free and Zero-Overhead Methods BEAVER (ours) 40.7 37.6 22.1 57.4 38.2 57.1 42.2 1,982 5x 3.0 5.2x 32.0 1,878 5x 2.5 4.9x

3,000-token constraint

(1) Unsupervised Statistical Methods Selective-Context 23.3 39.2 25.0 23.8 27.5 53.1 32.0 3,328 3x 50.6 0.3x 20.7 3,460 3x 54.2 0.2x

- LongLLMLingua 40.7 46.2 27.2 70.6 53.0 55.2 48.8 3,283 3x 10.0 1.6x 33.0 3,412 3x 8.2 1.5x

(3) Supervised and Specialized Learning Methods

SBERT 35.3 37.4 26.7 63.4 51.0 34.5 41.4 3,399 3x 7.7 2.0x 24.0 3,340 3x 5.9 2.1x OpenAI 34.5 38.6 26.8 63.4 49.6 37.6 41.7 3,421 3x 13.3 1.2x 22.4 3,362 3x 11.7 1.0x LLMLingua 31.8 37.5 26.2 67.2 8.3 53.2 37.4 3,421 3x 9.2 1.7x 30.7 3,366 3x 7.4 1.7x LLMLingua-2-small 35.5 38.1 26.2 67.5 23.9 60.0 41.9 3,278 3x 3.9 4.0x 33.4 3,089 3x 3.0 4.1x LLMLingua-2 35.5 38.7 26.3 69.6 21.4 62.8 42.4 3,392 3x 4.3 3.6x 33.5 3,206 3x 3.5 3.5x

(3) Training-free and Zero-Overhead Methods BEAVER (ours) 42.3 39.0 22.6 60.8 43.7 56.7 44.2 3,289 3x 3.5 4.5x 32.4 3,319 3x 2.9 4.2x

Reference Baselines Original Prompt 39.7 38.7 26.5 67.0 37.8 54.2 44.0 10,295 - 15.6 - 32.5 9,788 - 12.2 Zero-Shot 15.6 31.3 15.6 40.7 1.6 36.2 23.5 214 48x 1.6 9.8x 12.8 32 306x 1.0 12.2x

- Table 1: Performance comparison on LongBench and ZeroSCROLLS benchmarks with a 2,000 and 3,000-token budget. Best results are bolded, and second best are underlined. "Tokens" reports the average number of tokens remaining after compression, and "1/τ" is the average compression ratio. "Lat." and "Spd." denote Latency and Speedup, respectively.

our segment-page hierarchical design effectively minimizes inference overhead.

To evaluate the capability of locating specific information within large contexts, we conducted experiments on the RULER benchmark (Table 2). Existing compression methods suffer from severe performance degradation in this setting. For example, LLMLingua and LongLLMLingua fail to retrieve key information in multi-needle tasks and achieve only single-digit scores on S-2 and S-

- 3. In contrast, BEAVER exhibits strong robustness, maintaining 100.0% accuracy on all singleneedle tasks and achieving an average score of 83.7, which substantially surpasses the second-best method LLMLingua-2 (47.9). This capability is further corroborated by the Needle-in-a-Haystack visualizations in Appendix D. As shown in Figures 6 and 7, BEAVER attains near-perfect recall on Single-Needle (100%) and Multi-Needle (99%) tasks, whereas LongLLMLingua fails almost completely (< 10%) and LLMLingua-2 shows instability (86% and 72% respectively). This sharp contrast highlights the advantage of our PageEncoder and QueryPlanner, which alleviate the "lost in the

middle" issue (Liu et al., 2024) by preserving local semantic structure and leveraging in-context ITF to capture rare but crucial tokens that are often discarded by probability-based pruners.

To further assess the generalization ability of BEAVER under unseen distributions, we report results on the L-Eval benchmark in Table 3. We also extend this evaluation to the open-weights Qwen38B model in Appendix E to demonstrate robust generalization across model families. Despite being training-free, BEAVER demonstrates superior adaptability compared to learning-based baselines trained on extensive datasets. BEAVER achieved the highest average score of 57.6, outperforming LongLLMLingua (51.5) and LLMLingua-2 (54.6). BEAVER ranks first on 4 out of 6 datasets, with notable gains on SFictionQA and Legal Contract QA. These results confirm that our structure-aware sentence smoothing mechanism effectively preserves discourse coherence, which is crucial for complex reasoning tasks in domains such as law and literature. We provide concrete examples comparing the compression outputs of LongLLMLingua, LLMLingua-2, and BEAVER in Appendix F to il-

Single Multi FWE QA

Avg

Method

AVG

Tokens S-1 S-2 S-3 Key-1 Key-2 Key-3 Val Qry VT Freq QA-1 QA-2

LLMLingua 100.0 5.0 4.0 7.0 6.0 12.0 6.0 4.8 59.4 90.7 15.0 37.0 28.9 3,230 LongLLMLingua 100.0 6.0 4.0 9.0 2.0 3.0 7.8 7.3 60.4 91.3 19.0 36.0 28.8 3,179 LLMLingua-2 27.0 86.0 27.0 72.0 11.0 2.0 80.3 82.3 3.4 95.3 45.0 43.0 47.9 3,132 BEAVER (ours) 100.0 100.0 100.0 99.0 88.0 41.0 99.8 99.8 78.2 93.7 50.0 55.0 83.7 3,198

- Table 2: Performance of BEAVER and baselines on RULER (16k context, 3k budget). "S": Single-needle; "Multi": Multi-needle (Keys/Values) & Variable Tracking (VT); "FWE": Frequent Word Extraction; "QA": Question Answering.

Methods Coursera QA QuALITY SFictionQA TPO LongFQA Legal Contract QA AVG Tokens 1/τ LLMLingua 58.9 50.0 60.9 65.1 34.3 21.9 48.5 2,122 4× LongLLMLingua 62.2 51.0 65.6 71.0 37.7 21.3 51.5 2,196 4× LLMLingua-2 64.4 55.0 66.4 73.2 45.0 23.7 54.6 2,110 4× Simply-BEAVER 63.1 52.9 74.2 72.5 39.7 27.2 54.9 2,123 4× BEAVER (ours) 64.4 56.9 76.6 74.0 44.7 28.8 57.6 2,180 4×

Table 3: Out-of-domain evaluation on general long-context scenarios with a 2,000-token budget.

lustrate how token-level fragmented compression can disrupt narrative consistency.

##### 4.2.2 Efficiency Analysis

Efficiency is critical for real-world deployment, particularly as context lengths exceed 100k tokens. To evaluate the computational scalability, we conducted a latency analysis and compared with the LongLLMLingua and LLMLingua-2 families (including standard and small variants). Experiments were performed under different context lengths ranging from 16k to 128k, with the target output length fixed at 3,000 tokens.

Quantitative results are shown in Fig. 4. BEAVER exhibits a clear efficiency advantage across all context lengths. For a 128k-token ultra-long context, our method completes the compression in only 1.20 s, while the coarse-to-fine LongLLMLingua requires about 31.7 s, corresponding to a 26.4× speedup. Even compared with the highly optimized distilled models LLMLingua2 and its lightweight variant LLMLingua-2-small, BEAVER still achieves 5.9× and 2.3× speedup, respectively. Moreover, the latency of BEAVER grows linearly with a significantly flatter slope than all baselines. This suggests that our segment-page hierarchical design effectively maximizes parallelism, avoiding the computational bottlenecks typical of sequential processing.

##### 4.3 Ablation Study

To evaluate the effectiveness of each component, we conducted detailed ablations on the QA subset of LongBench. All experiments used a 2,000-token

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 4: Comparison of inference latency versus context length for different compression methods.

compression budget. Results are summarized in Table 4, covering segmentation granularity, PageEncoder representations, QueryPlanner scoring, and hierarchical page selection strategies.

Impact of Segmentation Granularity. We first examined the physical page size M. As shown in Table 4, using overly small pages (M = 32) reduces the average score by 1.8 points, since fine granularity fragments contiguous local contexts. Conversely, using overly large pages (M = 128) decreases performance by 2.4 points, indicating that coarse pages introduce substantial background noise, diluting the density of useful information.

Effectiveness of Encoder and Planner Designs. Regarding PageEncoder, removing max or mean pooling yields a ∼2.6 point drop, confirming their complementarity. Ablating the multi-vector query strategy causes a significant degradation (−2.9), suggesting that fine-grained query interactions are

Configuration S-Doc M-Doc Avg. ∆ BEAVER 40.7 37.6 39.2 – Segmenter

Page Size M=32 38.5 36.3 37.4 -1.8 Page Size M=128 37.5 36.1 36.8 -2.4 PageEncoder

w/o Max-Pooling 39.2 33.7 36.5 -2.7 w/o Mean-Pooling 39.2 34.0 36.6 -2.6 w/o Multi-Token Query 38.3 34.3 36.3 -2.9 LLaMA3-8B Embedding 39.4 38.2 38.8 -0.3

###### QueryPlanner

w/o ITF Score 35.7 37.3 36.5 -2.7 Semantic Only 36.5 29.9 33.2 -6.0 Lexical Only 38.8 33.4 36.1 -3.1 w/o Sentence Smooth 40.4 34.7 37.6 -1.6

Selection Policy Flow Only 33.3 28.8 31.1 -8.1 Anchor Only 15.0 20.0 17.5 -21.7 Flash Only 39.7 33.2 36.4 -2.7

- Table 4: Ablation study on LongBench QA (2K budget). ∆ denotes the performance drop relative to BEAVER.

critical for capturing complex constraints. We also verify the robustness of the backbone embedding model: replacing the default Qwen3-8B Embedding with Llama3-8B Embedding yields a negligible fluctuation (∆ = −0.3), indicating that our framework generalizes well across different model families. For QueryPlanner, removing lexical matching leads to the most severe drop (∆ = −6.0), highlighting the necessity of exact surface overlap. Removing semantic matching also incurs a 3.1 point loss. Furthermore, removing the in-context ITF score reduces performance by 2.7 points, validating that unsupervised context-aware weighting effectively suppresses high-frequency noise. Sentence smoothing provides a modest gain (+1.6), proving vital for repairing truncated sentence boundaries in multi-document tasks.

Significance of Hierarchical Selection. Finally, we analyzed the page selection strategy. Using only Flash achieves a score of 36.4 but remains below the full model. Relying solely on Flow causes a sharp drop to 31.1, indicating that local context alone is insufficient. Keeping only Anchor leads to a collapse in reasoning performance (17.5). These results validate the hybrid design of BEAVER: Flash retrieves highly relevant evidence, while Anchor and Flow ensure global instruction following and local coherence.

##### 4.4 Model Scalability Analysis

We evaluated the scalability of BEAVER on the Qwen3 series 0.6B–32B parameters. As shown in Fig. 5, BEAVER exhibits high stability across parameter scales compared to learning-based methods, with performance retention consistently be-

LLMLingua LongLLMLingua LLMLingua-2 Ours

120

| |
|---|

| |
|---|

RetentionRate(%ofDensePerformance)

| |
|---|

98%

100

Dense (Upper Bound)

88%

87%

84%

80

60

46% 48% 48%

42%

40

33%

32%

30% 31%

29% 31% 30%

28%

20

0

0.6B 4B 8B 32B

Qwen3 Model Size

Figure 5: Scalability and robustness analysis on RULER (16k context). Performance is normalized against the Dense upper bound (dashed line represents 100%).

tween 84%–98%. Notably, on the lightweight Qwen3-0.6B, BEAVER achieves 98% performance retention, significantly exceeding LLMLingua-2 at 42% and LongLLMLingua at 33%. Detailed results are available in Table 11. This consistent advantage across scales results from the training-free design. By utilizing intrinsic statistical signals instead of fitting specific data distributions, BEAVER effectively avoids overfitting. This independence from model scale enables BEAVER to serve as a plugand-play universal module for seamless integration into various LLM inference pipelines. More detailed analysis and discussion regarding these results are provided in the Appendix G.

#### 5 Conclusion

In this paper, we present BEAVER, a hardwareefficient and training-free prompt compression framework designed to overcome the computational and memory bottlenecks of long-context LLMs. By introducing a hierarchical segmentpage mechanism, we transform irregular sequence compression into optimized tensor operations. Our approach leverages unsupervised in-context ITF weighting and a structure-aware hybrid planner to effectively retain both global semantics and finegrained lexical details, avoiding the semantic collapse common in probability-based pruning. Experimental results confirm that BEAVER establishes SOTA performance in question answering and finegrained retrieval, while maintaining highly competitive fidelity in summarization tasks. Crucially, it offers superior scalability, achieving a 26.4× speedup on 128k contexts compared to learningbased baselines. As a plug-and-play module requiring no parameter updates, BEAVER provides

a practical and robust foundation for efficient largescale long-document understanding.

#### Limitations

Although BEAVER achieves a favorable balance between efficiency and performance, there are a few limitations. First, the page-level granularity prioritizes structural integrity and speed, but it is inherently less precise than fine-grained token pruning, which may occasionally result in the retention of minor redundancies within selected segments. Second, our retrieval mechanism relies on semantic and lexical similarity. It may face challenges in deep multi-hop reasoning scenarios where the supporting evidence shares little direct overlap with the query and requires intermediate deductive steps. Finally, as a training-free method, BEAVER depends on pre-set hyperparameters (e.g., weighting factors) that might require manual adjustment to achieve optimal results across drastically different domains, unlike end-to-end learning models that adapt automatically.

#### Ethical Considerations

This work introduces a prompt compression framework, BEAVER, designed to improve the efficiency of Long Context LLMs. While our method aims to reduce computational costs, there are potential risks associated with information pruning.

The primary risk of any compression method, including BEAVER, is the potential loss of critical qualifiers or safety constraints (e.g., negations or warnings) present in the original context. In sensitive domains such as legal or medical analysis (as evaluated in our experiments on L-Eval), such omission could lead to factually incorrect or unsafe downstream generations. Although our structureaware design and sentence smoothing aim to mitigate fragmentation, users should exercise caution and verify outputs when applying this method to high-stakes decision-making tasks.

BEAVER relies on off-the-shelf embedding models (e.g., Qwen) to calculate page importance. Biases inherent in these pre-trained models, such

- as the under-representation of certain dialects or cultural contexts, may be propagated. This could result in the systematic filtering of text from minority groups if the embedding model deems such patterns as "low importance" or "redundant." Future work should investigate the fairness of relevance scoring across diverse linguistic demographics.

#### References

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-eval: Instituting standardized evaluation for long context language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14388–14411.

Anthropic. 2025. Introducing claude sonnet 4.5. Accessed: 2025-12-27.

Simon A Aytes, Jinheon Baek, and Sung Ju Hwang. 2025. Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching. arXiv preprint arXiv:2503.05179.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, and 1 others. 2024. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119– 3137.

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems.

Google DeepMind. 2025. Gemini 3 pro model card.

Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, ZhongZhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, and 1 others. 2025. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1135–1159.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. 2021. Efficient attentions for long document summarization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1419–1436.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024.

Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1658–1677.

Philip N Johnson-Laird. 2010. Mental models and human reasoning. Proceedings of the National Academy of Sciences, 107(43):18243–18250.

Hoyoun Jung and Kyung-Joong Kim. 2024. Discrete prompt compression with reinforcement learning. IEEE Access, 12:72578–72587.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39– 48.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Jia Li, Xuyuan Guo, Lei Li, Kechi Zhang, Ge Li, Zhengwei Tao, Fang Liu, Chongyang Tao, Yuqi Zhu, and Zhi Jin. 2025a. Longcodeu: Benchmarking longcontext language models on long code understanding. arXiv preprint arXiv:2503.04359.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2024. Loogle: Can long-context language models understand long contexts? In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16304–16333.

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023a. Compressing context to enhance inference efficiency of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 6342–6353.

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023b. Compressing context to enhance inference efficiency of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6342–6353.

Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Collier. 2025b. Prompt compression for large language models: A survey. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7182–7195.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, WeiMing Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. Awq: Activation-aware weight quantization for ondevice llm compression and acceleration. In Proceedings of machine learning and systems, volume 6, pages 87–100.

Barys Liskavets, Maxim Ushakov, Shuvendu Roy, Mark Klibanov, Ali Etemad, and Shane K Luke. 2025. Prompt compression with context-aware sentence encoding for fast and improved llm inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24595–24604.

Junyi Liu, Liangzhi Li, Tong Xiang, Bowen Wang, and Yiming Qian. 2023. Tcra-llm: Token compression retrieval augmented large language model for inference cost reduction. arXiv preprint arXiv:2310.15556.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, and 1 others. 2024. Mmlongbenchdoc: Benchmarking long-context document understanding with visualizations. In Advances in Neural Information Processing Systems, volume 37, pages 95963–96010.

OpenAI. 2024. Embeddings - openai api. Accessed: 2025-01-13.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, and 1 others. 2024. Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In arXiv preprint arXiv:2403.12968.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. Zeroscrolls: A zeroshot benchmark for long text understanding. arXiv preprint arXiv:2305.14196.

Shivam Shandilya, Menglin Xia, Supriyo Ghosh, Huiqiang Jiang, Jue Zhang, Qianhui Wu, Victor Rühle, and Saravan Rajmohan. 2025. Taco-rl: Task aware prompt compression optimization with reinforcement learning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1582–1597.

Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, and Jonas Geiping. 2025. The illusion of diminishing returns: Measuring long horizon execution in llms. arXiv preprint arXiv:2509.09677.

Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, and Zhiming Zheng. 2024. Adacomp: Extractive context compression with adaptive predictor for retrieval-augmented large language models. arXiv preprint arXiv:2409.01579.

Lu Zhao, Rong Shi, Shaoqing Zhang, Shangchao Su, Ziqing Yin, Zhiyan Cui, Hongfeng Sun, Baoguo He, Yueqiang Chen, Liang Dong, and 1 others. 2025. Mofa: A unified performance modeling framework for llm pretraining. arXiv preprint arXiv:2511.09837.

#### A Datasets

LongBench (Bai et al., 2024) serves as the first bilingual multitask benchmark designed to evaluate the long context understanding capabilities of Large Language Models (LLMs) on texts ranging from 1k to 22k tokens. As shown in Table 5, the dataset comprises 21 subdatasets across six task categories involving Chinese, English, and code, totaling 4,750 test samples. All subsets adhere to a standardized input and output format to ensure consistent evaluation across tasks.

ZeroSCROLLS (Shaham et al., 2023) is a zero shot benchmark designed to evaluate natural language understanding capabilities over long texts. It comprises test sets from ten distinct tasks, each requiring reasoning over various types of long content. Unlike traditional benchmarks, ZeroSCROLLS excludes training data to focus on pure zero shot evaluation, requiring models to demonstrate generalization without task specific fine tuning. As shown in Table 6, the benchmark covers diverse capabilities such as query based summarization, question answering, multiple choice reasoning, and information aggregation. Each task features natural long texts, with an average document length of approximately 10,000 words.

RULER (Hsieh et al., 2024) is a high quality synthetic benchmark designed to evaluate the long context modeling capabilities of language models. Constructed with flexible configurations, it allows for customizable sequence lengths and task complexities to enable comprehensive evaluation across varying context window requirements. The benchmark comprises 13 representative tasks spanning four distinct categories: (i) Retrieval Tasks evaluate the ability to locate specific information and extend beyond the traditional Needle In A Haystack paradigm by covering various needle types and

quantities; (ii) Multi hop Tracing Tasks require tracing reasoning chains across segments to test the capacity for maintaining and connecting information throughout long sequences; (iii) Aggregation Tasks assess the capability to collect, summarize, and synthesize information from distributed locations; and (iv) Question Answering Tasks evaluate comprehensive understanding via natural language questions that require synthesizing information from diverse parts of the context. Detailed statistics for RULER are provided in Table 7.

L-Eval (An et al., 2024) L-Eval is a comprehensive benchmark for long context language models designed for the standardized evaluation of processing and reasoning on long inputs. The dataset is constructed through rigorous human annotation and quality inspection to ensure high reliability. It includes 411 documents from diverse domains such as law, finance, academia, and fiction. With an average document length of 7,217 tokens and a total corpus of approximately 60,000 tokens, it is suitable for evaluating long context understanding. The benchmark contains 2,197 manually verified query and answer pairs across 20 tasks. Detailed statistics are presented in Table 8.

#### B Baseline Models

##### B.1 Unsupervised Statistical Methods

Selective-Context (Li et al., 2023b) proposes a context compression technique designed to enhance LLM inference efficiency by identifying and pruning redundant input information. The method employs a base causal language model to evaluate the self-information of lexical units (such as sentences or tokens), selectively retaining content with higher information values while discarding predictable redundancy.

LongLLMLingua (Jiang et al., 2024) addresses the challenges of high computational cost, noise, and position bias inherent in long-context scenarios. Building upon LLMLingua, it introduces a question-aware coarse-to-fine compression method to enhance the density of key information relative to the user’s query. Furthermore, the method incorporates a document reordering strategy to mitigate the "lost-in-the-middle" phenomenon and employs a post-compression subsequence recovery mechanism to preserve the integrity of critical details.

###### Task Category # Subsets Language Avg. Length (tokens) Metric # Samples Representative Datasets

Single-Doc QA 4 3 En, 1 Zh 3.6k – 18.4k F1 750 NarrativeQA, Qasper, MultiFieldQA Multi-Doc QA 4 3 En, 1 Zh 4.9k – 15.8k F1 / Rouge-L 800 HotpotQA, DuReader, MuSiQue Summarization 5 4 En, 1 Zh 2.1k – 15.4k Rouge-L 1,000 GovReport, VCSUM, SAMSum Few-Shot Learning 3 2 En, 1 Zh 5.2k – 22.3k Acc. / F1 600 TREC, LSHT, TriviaQA Synthetic Tasks 3 2 En, 1 Zh 6.7k – 11.1k Accuracy (EM) 600 PassageCount, PassageRetrieval Code Completion 2 Multi-Code 1.2k – 4.2k Edit Sim. 1,000 LCC, RepoBench-P

Total 21 Bi + Code 1.2k – 22.3k — 4,750 —

- Table 5: Statistics of LongBench six major task categories. Here, "En" denotes English datasets, and "Zh" denotes Chinese datasets.

Task Category # Subsets Language Avg. Length (words) Metric # Samples Representative Datasets

Summarization 4 En 4.9k – 10.8k Rouge 1,378 GovReport, QMSum, SQuALITY Question Answering 4 En 1.7k – 49.4k F1 / Accuracy 2,000 NarrativeQA, Qasper, MuSiQue Aggregation 2 En 5.5k – 6.8k ES / Cidx 1,000 SpaceDigest, BookSumSort

Total 10 En 1.7k – 49.4k — 4,378 —

- Table 6: Statistics of ZeroSCROLLS task categories. The benchmark consists of 10 datasets across 3 major categories. "ES" denotes Exponential Similarity, and "Cidx" denotes Concordance Index.

B.2 Supervised and Specialized Learning Methods

LLMLingua (Jiang et al., 2023) presents a coarse-to-fine prompt compression framework designed to accelerate inference by pruning prompts while preserving semantic integrity. The method employs a budget controller to dynamically allocate compression ratios across different prompt components and utilizes a token-level iterative algorithm to capture conditional dependencies between tokens. Additionally, it incorporates instruction tuning to align the distribution of the smaller compression model with the target LLM.

Sentence-BERT (Reimers and Gurevych, 2019) modifies the pretrained BERT architecture by employing siamese and triplet network structures to derive semantically meaningful sentence embeddings. This framework allows sentences to be compared using cosine similarity, thereby significantly reducing the computational overhead associated with pair-wise cross-encoding while maintaining high performance on semantic search and clustering tasks.

LLMLingua-2 (Pan et al., 2024) advances taskagnostic prompt compression by framing the problem as token classification rather than causal entropy-based pruning. The authors distill knowledge from GPT-4 to create a high-quality extractive compression dataset, which is then used to train a bidirectional Transformer encoder (e.g., XLMRoBERTa, mBERT) to classify tokens as "preserve" or "discard". This approach enables the

model to leverage full bidirectional context for better information retention, ensuring faithfulness to the original text while achieving significantly lower latency compared to autoregressive or causal-based methods.

#### C Detailed Analysis on ZeroSCROLLS

Table 9 reports the detailed results on the ZeroSCROLLS benchmark. Overall, our proposed BEAVER framework demonstrates strong robustness across all ten subtasks, achieving average scores of 32.0 and 32.4 under the compression budgets of 2,000 and 3,000 tokens, respectively. This performance substantially surpasses unsupervised statistical baselines such as Selective-Context and achieves results comparable to LongLLMLingua. Notably, although BEAVER is entirely trainingfree, it matches or even exceeds state-of-the-art supervised methods such as LLMLingua-2 across metrics. For example, it maintains highly competitive scores on GovReport and SQuALITY while avoiding the costly overhead of model fine-tuning. On detail-sensitive tasks such as Qasper and QMSum, BEAVER still attains high accuracy, largely because it can identify key cues such as entities and numbers that are often overlooked in purely semantic spaces. Moreover, on tasks requiring long-range dependency understanding, such as NarrativeQA, the model achieves a high score of 34.0. This result supports the necessity of the structural priors introduced in our QueryPlanner, where retaining anchor pages and local streaming context effectively preserves discourse coherence and causal structure.

###### Task Category # Subsets Language Length (tokens) Metric # Samples Representative Datasets

Retrieval (NIAH) 8 En 4k – 128k Accuracy Variable S-NIAH, MK-NIAH, KV-Ret Multi-hop Tracing 1 En 4k – 128k Accuracy Variable Variable Tracking (VT) Aggregation 2 En 4k – 128k Accuracy Variable CWE, FWE Question Answering 2 En 4k – 128k EM / F1 Variable SQuAD, HotpotQA

Total 13 En 4k – 128k — Variable —

- Table 7: Statistics of RULER task categories. RULER contains 13 configurable tasks across 4 categories. The context length is synthetic and typically evaluated from 4k to 128k tokens. "NIAH" denotes Needle In A Haystack tasks.

Task Category # Subsets Language Avg. Length (tokens) Metric # Samples Representative Datasets

Summarization 7 En 7.3k – 20.0k Rouge 386 GovReport, QMSum, SPACE Open-Ended QA 6 En 3.9k – 62.3k Rouge / F1 764 NarrativeQA, NQ, Qasper Multiple Choice / Exam 6 En 3.9k – 16.4k Accuracy 957 TOEFL, QuALITY, Coursera Code Understanding 1 En 31.6k Accuracy 90 CodeU

Total 20 En 3.9k – 62.3k — 2,197 —

- Table 8: Statistics of L-Eval Task Categories. The suite comprises 20 datasets categorized by task type. "En" denotes English language tasks.

#### D Needle-in-a-Haystack Visualization

To provide a comprehensive evaluation of context modeling capabilities, we visualize the retrieval robustness of BEAVER compared to baselines using the Needle-in-a-Haystack test on the RULER benchmark. We conduct experiments under a highcompression setting. Specifically, the original context length is fixed at 16k tokens, while the compressed token constraint is strictly limited to 3,000 tokens (approximately 5.3× compression). We evaluate both Single-Needle and Multi-Needle retrieval tasks across 100 distinct samples. All inference is performed using gpt-3.5-turbo as the backend model to ensure consistent evaluation.

Figures 6 and 7 present the retrieval heatmaps, where green blocks indicate successful retrieval and red blocks denote failure. The results demonstrate the superior robustness of BEAVER, which achieves near-perfect performance by maintaining a "full green" heatmap with 100.0% and 99.0% accuracy in Single-Needle and Multi-Needle retrieval, respectively. This indicates that BEAVER effectively preserves semantic integrity even under aggressive compression. In contrast, LLMlingua-2 exhibits degradation in complex scenarios; while retaining acceptable performance in the single-needle task (86.0%), its accuracy declines significantly to 72.0% in the multi-needle setting, reflecting an inability to consistently retrieve distributed information. Furthermore, LongLLMlingua displays severe information loss characterized by predom-

inantly red heatmaps, achieving only 6.0% and 9.0% accuracy for single and multi-needle tasks, which suggests it struggles to prioritize critical query-relevant information within the 3,000-token budget.

#### E Additional Evaluation on L-Eval Benchmark

To further verify the generalization capability of BEAVER on open-weights models beyond the ZeroSCROLLS and RULER benchmarks, we conducted an out-of-domain evaluation on the L-Eval suite (An et al., 2024) using Qwen3-8B3. L-Eval provides a diverse set of long-context tasks including academic QA, legal analysis, and fiction reading, which serve as a robust testbed for real-world applicability.

The results are presented in Table 10. Consistent with our observations on GPT-series models, BEAVER demonstrates superior performance on Qwen3-8B, achieving the highest average score of 45.1. Our method outperforms the strongest baseline, LLMLingua-2, by approximately 1.6 points on average. Notably, in complex reasoning tasks such as TPO (Topology) and Legal Contract QA, BEAVER leads by significant margins (+1.9 and +0.6 respectively), indicating that our structure-aware compression better preserves logical dependencies required for specialized domains. All methods operate under a similar com-

3https://huggingface.co/Qwen/Qwen3-8B

ZeroSCROLLS Datasets Efficiency

Methods

GvRp SSFD QMsm SQAL QALT Nrtv Qspr MuSQ SpDg BkSS AVG Tokens 1/τ Lat. Spd.

- 2,000-token constraint

- (1) Unsupervised Statistical Methods

- Selective-Context 19.0 8.4 9.7 12.4 47.0 12.5 21.6 11.5 41.2 11.0 19.4 1,865 5x 47.5 0.3x LongLLMLingua 20.1 12.4 14.9 16.5 65.1 27.7 30.7 23.6 68.5 47.2 32.7 1,753 6x 5.2 2.3x

(2) Supervised and Specialized Learning Methods

SBERT 10.2 7.9 13.7 13.2 60.0 8.1 10.8 1.7 37.2 42.8 20.5 1,773 6x 4.1 3.0x OpenAI 11.1 8.0 11.8 13.6 60.0 7.1 13.2 4.0 33.6 43.6 20.6 1,784 5x 9.9 1.2x LLMLingua 19.4 11.9 13.1 16.0 62.1 23.7 24.0 22.4 33.9 44.9 27.2 1,862 5x 4.8 2.5x

- LLMLingua-2-small - - - - - - - - - - 33.3 1,862 5x 2.6 4.7x

- LLMLingua-2 - - - - - - - - - - 33.4 1,898 5x 3.0 4.1x

(3) Training-free and Zero-Overhead Methods BEAVER (ours) 17.2 12.5 15.2 16.5 71.4 34.0 34.4 20.2 56.8 41.8 32.0 1,878 5x 2.5 4.9x

3,000-token constraint

- (1) Unsupervised Statistical Methods

Selective-Context 20.8 9.1 11.7 13.4 50.0 9.8 26.1 11.0 46.0 9.5 20.7 3,460 3x 54.2 0.2x LongLLMLingua 22.1 12.8 15.9 17.1 67.0 27.8 31.3 23.9 65.8 46.5 33.0 3,412 3x 8.2 1.5x

- (2) Supervised and Specialized Learning Methods SBERT 16.5 9.8 12.3 15.2 60.0 14.6 23.4 12.1 39.4 36.4 24.0 3,340 3x 5.9 2.1x OpenAI 14.3 8.3 12.0 15.3 66.7 13.3 24.3 11.7 31.2 26.4 22.4 3,362 3x 11.7 1.0x LLMLingua 18.7 10.0 14.9 16.8 61.9 26.9 27.2 23.4 62.9 44.5 30.7 3,366 3x 7.4 1.7x

LLMLingua-2-small - - - - - - - - - - 33.4 3,089 3x 3.0 4.1x LLMLingua-2 - - - - - - - - - - 33.5 3,206 3x 3.5 3.5x

- (3) Training-free and Zero-Overhead Methods BEAVER (ours) 19.4 12.5 15.9 17.5 70.6 34.0 34.5 21.8 56.0 41.2 32.4 3,319 3x 2.9 4.2x

Reference Baselines Original Prompt 21.8 12.1 17.9 17.4 66.7 25.3 29.8 20.0 69.7 44.1 32.5 9,788 - 12.2 Zero-Shot 9.4 3.0 8.6 11.4 42.9 10.6 12.4 5.5 4.2 0.0 12.8 32 306x 1.0 12.2x

- Table 9: Detailed evaluation on ZeroSCROLLS datasets. We report the performance on individual sub-tasks: GovReport (GvRp), SummScreenFD (SSFD), QMSum (QMsm), SQuALITY (SQAL), Quality (QALT), NarrativeQA (Nrtv), Qasper (Qspr), MuSiQue (MuSQ), SpaceDigest (SpDg), and BookSumSort (BkSS). "Lat." and "Spd." denote Latency and Speedup, and "-" indicates that the results were not available from the original paper.

pression ratio (approx. 4×, reducing context to 2,000 tokens). BEAVER achieves this performance gain without the need for external training (unlike LLMLingua-2) or unstable perplexity calculations (unlike LongLLMLingua), highlighting its effectiveness as a lightweight, plug-and-play solution for open-source LLMs.

#### F Qualitative Comparison with Baseline Methods

To provide an intuitive assessment of our method under strict inference constraints, we conducted qualitative analyses on four representative tasks from the L-Eval benchmark (An et al., 2024), including code generation (CodeU), government report summarization (GovReport (Huang et al., 2021)), financial question answering (LongFQA), and few shot mathematical reasoning (GSM100). We impose strict length budgets ranging from 500

to 1500 tokens to evaluate each method’s ability to preserve critical information under extreme compression. We compared BEAVER with unsupervised statistical methods (for example, LongLLMLingua (Jiang et al., 2024)) and supervised methods (for example, LLMLingua 2 (Pan et al., 2024)), and focus on failure modes related to semantic noise, program logic, chain of thought structure, and code syntax.

First, in LongFQA with substantial semantic noise under a 500 token budget, the model must extract the pricing logic of iPhone 15 Pro from context mixed with irrelevant inventory information about consumer electronics such as NVIDIA GPUs and Dyson vacuum cleaners (Figure 8). Results show that the statistical baseline (Baseline A) over compresses and corrupts critical surface forms, for example deleting the key surcharge term “20%” into “a2%”, which breaks the numerical logic. The

Qwen3-8B

Methods

Tokens↓ 1/τ↑ Coursera QA QuALITY SFictionQA TPO LongFQA Legal Contract QA AVG

LLMLingua 24.9 48.0 60.9 66.6 13.6 11.2 37.5 2,122 4× LongLLMLingua 30.2 49.0 69.5 72.9 15.2 10.9 41.3 2,196 4× LLMLingua-2 25.3 53.0 74.2 78.4 16.9 12.8 43.5 2,110 4×

BEAVER (ours) 28.3 57.4 73.4 80.3 17.7 13.4 45.1 2,180 4×

- Table 10: Out-of-domain evaluation on general long-context scenarios (L-Eval) using Qwen3-8B. All methods operate under a 2,000-token input budget constraint. Highest scores are bolded, and second highest are underlined.

[Figure 18]

[Figure 19]

[Figure 20]

###### (A) BEAVER (B) LLMlingua-2 (C) LongLLMlingua

Figure 6: Visualization of Single-Needle Retrieval Accuracy.

supervised baseline (Baseline B) produces fluent text but fails to preserve the causal relation between the base price and the surcharge rule, leading to hallucination and an incorrect output of $899. In contrast, our method uses a hybrid semantic and lexical scoring mechanism to identify and retain the price table and the surcharge memo scattered across the document, while filtering the intervening noise blocks. This enables the downstream model to correctly infer the final price of $1,078.80, demonstrating strong robustness to distraction.

Second, for the GovReport summarization task under a 500 token budget, which tests long document coherence and logical completeness, the goal is to extract the procedural logic of the “72 hour rule” and its exemption conditions (Figure 9). Baselines exhibit severe semantic fragmentation. The statistical method damages syntactic structure, for example generating corrupted fragments such as “Rements R”, which prevents rule extraction. The supervised method retains keywords such as “one day layover” but loses the connecting logic that triggers the exemption, namely “convening two legislative days”, making the summary factually incomplete. By contrast, our method leverages a Segmenter to respect natural language boundaries, preserving complete sentence structure for both the rule and its exceptions, and yields a coherent and accurate summary.

Third, on the GSM100 few shot reasoning task

with a 1500 token budget, we evaluate whether compression preserves chain of thought structure (Figure 10). Baselines often treat few shot demonstrations as unstructured text and destroy the boundaries between question, reasoning, and answer, causing the large model to miss the in context examples and degrade to zero shot guessing, which fails on nontrivial arithmetic. Our method treats each example as a coherent page unit and preserves the chain of thought steps of highly relevant demonstrations, such as “Let’s think step by step...”, enabling genuine in context learning and guiding the model to the correct result 2 + 1 = 3.

Finally, in the CodeU code task under a 1000 token budget, where syntactic correctness is critical, baselines struggle more severely (Figure 11). The statistical method indiscriminately removes tokens deemed redundant, including structural delimiters, yielding unparsable code such as “importLib np..”. The supervised method can collapse, compressing complex logic into meaningless punctuation. Given the high information density of code, our method prioritizes rare tokens and structural separators through the lexical pathway in PageEncoder, preserving function definitions and the demonstration examples in docstrings. This allows the model to pass syntax checks and to learn from examples to correctly predict execution outcomes.

[Figure 21]

[Figure 22]

[Figure 23]

###### (A) BEAVER (B) LLMlingua-2 (C) LongLLMlingua

Figure 7: Visualization of Multi-Needle Retrieval Accuracy.

#### G Additional Analysis on Scalability and Robustness

In this section, we provide a detailed analysis of the scalability results presented in Table 11, examining the performance of BEAVER across the Qwen3 series (0.6B to 32B) in comparison to the two categories of baselines defined in the related work: Unsupervised Statistical Methods and Supervised Specialized Learning Methods.

- G.1 Analysis vs. Unsupervised Statistical Methods

Unsupervised methods like LongLLMLingua rely on causal entropy or perplexity metrics to identify key information. While these metrics are effective for larger models, our experiments reveal their fragility on lightweight architectures. As shown in Table 11, LongLLMLingua achieves an average score of only 25.9 on Qwen3-0.6B. This performance drop indicates that probability distributions in smaller models are significantly noisier and less reliable as proxies for semantic importance compared to their larger counterparts. Furthermore, the "question-aware" mechanism in LongLLMLingua depends on the model’s zero-shot understanding of the query, which is inherently limited in 0.6Bscale models. In contrast, BEAVER utilizes intrinsic attention statistics, achieving a score of 76.8 on the same 0.6B model. This substantial margin demonstrates that attention-based signals are far more robust and invariant to model scale than the perplexity-based signals used by other unsupervised baselines.

- G.2 Analysis vs. Supervised and Specialized Learning Methods

For supervised methods such as LLMLingua and LLMLingua-2, the primary bottleneck on smaller models stems from the distribution mismatch be-

tween the compression module and the target inference model. LLMLingua-2, for instance, employs a BERT-based encoder trained on data distilled from GPT-4. While this effectively captures information relevant to GPT-4, these patterns do not transfer well to the Qwen3-0.6B model (score: 32.9). The tokens considered "informative" by a GPT-4-aligned compressor are often too complex or distinct from the reasoning paths required by a lightweight model. Similarly, LLMLingua’s instruction tuning strategy fails to align effectively when the target model lacks sufficient capacity, resulting in the lowest score of 25.1. BEAVER circumvents these compatibility issues by design. By deriving compression decisions directly from the specific model instance being used, BEAVER ensures that the retained information is inherently aligned with that specific model’s processing mechanisms. This "self-adaptive" characteristic allows BEAVER to maintain 98% performance retention on the 0.6B model without the need for external training or domain adaptation.

#### H LLM Usage

We acknowledge the use of LLMs to assist with the writing process of this manuscript. Specifically, these tools were employed solely for the purpose of refining textual expression, correcting grammatical errors, and improving the flow of the narrative.

We explicitly state that:

- • The conceptualization, methodology design, and analysis of experimental results are entirely the original work of the authors.
- • No part of the experimental code, data processing pipelines, or implementation details was generated by AI assistants.

Single Multi FWE QA

Method

AVG S-1 S-2 S-3 Key-1 Key-2 Key-3 Val Qry VT Freq QA-1 QA-2

Qwen3-0.6b Dense 100.0 100.0 100.0 95.0 84.0 45.0 82.8 94.0 86.4 90.7 34.0 28.0 78.3

- LLMLingua 100.0 5.0 4.0 7.0 5.0 10.0 6.0 4.8 45.2 91.0 3.0 20.0 25.1

- LongLLMLingua 100.0 6.0 4.0 10.0 1.0 3.0 7.5 7.0 56.6 92.0 3.0 21.0 25.9 LLMLingua-2 27.0 69.0 24.0 57.0 7.0 2.0 44.0 45.8 0.6 92.0 10.0 17.0 32.9 BEAVER 100.0 100.0 100.0 97.0 88.0 40.0 97.5 97.3 58.6 89.7 25.0 29.0 76.8

Qwen3-4b

- Dense 100.0 100.0 100.0 100.0 99.0 100.0 100.0 100.0 99.6 95.7 74.0 55.0 93.6

LLMLingua 100.0 5.0 4.0 7.0 6.0 12.0 6.0 4.8 54.2 95.7 12.0 23.0 27.5

- LongLLMLingua 100.0 6.0 4.0 9.0 2.0 3.0 7.8 7.0 67.8 94.3 11.0 23.0 27.9 LLMLingua-2 27.0 77.0 27.0 77.0 11.0 2.0 59.8 73.3 0.8 94.0 39.0 29.0 43.1 BEAVER 100.0 100.0 100.0 100.0 88.0 42.0 99.8 99.8 71.8 89.7 43.0 54.0 82.3

Qwen3-8b

Dense 100.0 100.0 100.0 100.0 100.0 98.0 99.0 100.0 100.0 93.3 74.0 56.0 93.4 LLMLingua 100.0 5.0 4.0 7.0 6.0 12.0 6.0 4.8 54.6 97.0 20.0 28.0 28.7

- LongLLMLingua 100.0 6.0 4.0 10.0 1.0 3.0 8.0 7.3 62.6 96.3 20.0 29.0 28.9 LLMLingua-2 27.0 84.0 27.0 74.0 11.0 2.0 68.5 73.5 1.8 93.3 40.0 31.0 44.4 BEAVER 100.0 100.0 100.0 100.0 84.0 32.0 100.0 98.8 62.8 79.0 41.0 46.0 78.6

Qwen3-32b

Dense 100.0 100.0 100.0 100.0 100.0 100.0 99.5 100.0 100.0 97.7 86.0 60.0 95.3 LLMLingua 100.0 5.0 4.0 7.0 6.0 12.0 6.0 4.8 60.0 97.3 18.0 28.0 29.0

- LongLLMLingua 100.0 6.0 4.0 10.0 1.0 3.0 8.0 7.3 36.0 96.3 22.0 28.0 26.8 LLMLingua-2 27.0 82.0 27.0 74.0 9.0 2.0 68.0 75.8 3.2 94.7 49.0 35.0 45.6 BEAVER 100.0 100.0 100.0 100.0 88.0 41.0 100.0 99.8 70.0 90.0 51.0 55.0 82.9

- Table 11: Main Results on RULER benchmark (16k context). Highest scores are bolded, and second highest are underlined.

QUERY&ANSWER TASK

Query: "What is the price for an ordinary user to purchase an iPhone 15 Pro (256GB)?"

Context: Global Pricing Memorandum (Contains: Surcharge Rules, NVIDIA, Dyson, Cameras, Phones, etc.) ✅Ground Truth: $1,078.80 ($899 + 20% Surcharge)

- (A) UNSUPERVISED STATISTICAL METHODS (E.G., LONGLLMLINGUA) Q Global Pr ... Cru AdjustmentV: all regular customers-, or2 membersatives mand to a2% surcharge to the branchs logAutumn. [...Garbled Noise: DellEdgeE90ated, Sonypha 1, Dyson cleaner...]

... iPhone15 Pro) , is listed at $899. Please note this immediate ... Nov 12th.

MODEL ANSWER: "The document does not provide specific pricing information for the iPhone 15 Pro (256GB)."

Result: ❌FAILED (Syntax Corrupted: "20%" → "a2%", logic lost)

- (B) SUPERVISED AND SPECIALIZED LEARNING METHODS (E.G., LLMLINGUA-2) Pricing Strategy Inventory Memorandum. Adjustment Non-VIPs 20% surcharge prices. audit.

iPhone 15 Pro (256GB $899. subsidy carrier. iPhone 15 Pro Max (512GB $1,199.

MODEL ANSWER: "The price for an ordinary user to purchase an iPhone 15 Pro (256GB) is $899."

Result: ⚠HALLUCINATION (Missed logic connection due to entity noise)

[...Retained Irrelevant Entities: NVIDIA H100, Dell PowerEdge, Lenovo, Sony Alpha 1, Dyson Gen5, Roborock S8...]

- (C) TRAINING-FREE AND ZERO-OVERHEAD METHODS (OURS) Important Notice: For all regular customers... sales representatives are mandatory to add a 20% surcharge to the listed prices below.

[...Block Compression: Deleted 14 unrelated product categories (NVIDIA, Cameras, Vacuums) to preserve context window...]

For the iPhone 15 Pro (256GB, Natural Titanium), the device is listed at $899. Please note this price reflects a subsidy... Note: This promotion ended on Nov 12th.

MODEL ANSWER: "The price is $899 plus a 20% surcharge, making it $1,078.80."

Result: ✅CORRECT (Preserved Syntactic Structure & Critical Logic)

Figure 8: Qualitative comparison on Financial QA under a 500 token budget.

SUMMARIZATION TASK

Task: "Please help me summarize this government report."

Context: House Rules on Availability (72-hour rule, Layover requirements, and Waivers).

###### ✅Ground Truth Summary: Requires 72-hour availability. Waivers possible via: (1) Suspension, (2) Special Rule (one-day layover), (3) Convening 2nd legislative day.

- (A) UNSUPERVISED STATISTICAL METHODS (E.G., LONGLLMLINGUA) Rements R rules of the House... grant an to legisl by ning the time the must be being considered...

Under House draft and unedills and considered they " electronic at on Administration."

...Conference 8 House rule its jointatory statement... ly:// ) In report ording the,hour during session. is.ment usually notment This. a even ne...

... In, House can the for any a specified is Second the any measure.

MODEL ANSWER: "The House of Representatives rules outline procedures... allow for same-day rules to expedite the process."

Result: ❌VAGUE. Syntax destruction ("Rements R", "unedills") prevented the LLM from extracting specific rules like "72 hours" or "waiver methods".

- (B) SUPERVISED AND SPECIALIZED LEARNING METHODS (E.G., LLMLINGUA-2) Availability Requirements House Rules. rules availability reports special rules. reports bills electronic designated Committee Administration. Conference reports.

available 72 hours. text. Rule XI 2 two days submit. majority draft 72-hour. Bills resolutions 72 hours. waiving availability requirements.

last six days session... last three days session. rarely agrees adjournment.

Waiving Rules Committee 72-hour availability... same day... one-day layover. subject layover adopted day.

MODEL ANSWER: "...Bills must be available for 72 hours... Special rules can waive availability with a one-day layover..."

Result: ⚠FRAGMENTED. Retained keywords ("72 hours", "waiving") but lost the connective logic explaining *how* to waive (e.g., "convening two legislative days").

- (C) TRAINING-FREE AND ZERO-OVERHEAD METHODS (OURS) Alternatively, the conference report can be considered if it has been made publicly available in electronic form...

According to the rule, this 72-hour availability requirement does not apply during the last six days of a session.

[...Block Compression of General Rules...]

To waive the one-day layover requirement... the Rules Committee may report a special rule... If such a special rule is adopted, the House can then consider... a measure later on the same legislative day.

Although the House rarely chooses to do so, it could agree to call up... a measure in a single calendar day by convening two legislative days...

MODEL ANSWER: "...includes exceptions for the last six days... mentions the possibility of considering a measure in a single calendar day by convening two legislative days."

Result: ✅COHERENT. Successfully preserved complex procedural logic (Waivers, Two Legislative Days) by maintaining sentence integrity.

Figure 9: Qualitative comparison on GovReport summarization under a 500 token budget.

FEW-SHOT TASK

Query: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"

Context: Few-Shot Prompt containing 15 Math Word Problems with Chain-of-Thought (CoT) solutions. ✅Ground Truth: 3 (2 blue + 1 white)

- (A) UNSUPERVISED STATISTICAL METHODS (E.G., LONGLLMLINGUA) Question: Four with32 teeth went for checkup realizing they were having severe pain different numbers ofaged teeth... The person/4 of all his teeth removed... The answer is0 Question: Emily raised $10 by bies... The answer is 150 Solo to read 4book 20 pages... 4 (Science) + 20 (cial + 7) + 8) = 9 pages... The answer is 6 [...Garbled Syntax: "The answer is0", "pies", "with32"]

MODEL ANSWER: "To find the total number of bolts needed for the robe... 2 bolts of blue fiber + 1 bolt of white fiber, which equals 3 bolts in total."

Result: ⚠PSEUDO-CORRECT (Zero-shot Reliance). The context is corrupted into noise. The LLM ignored it and relied entirely on internal pre-trained knowledge to guess this simple query. Fails on complex reasoning tasks.

- (B) SUPERVISED AND SPECIALIZED LEARNING METHODS (E.G., LLMLINGUA-2) Four adults 32 teeth severe tooth pain. different damaged teeth each removed. first 1/4 second 3/8... 12+8 = 20. third half 1/2*32

= 16. Emily raised $210 charity... 3 cookies 5 brownies. $2 * 3 $6 $3 * 5 = $15 brownies. [...Keyword Salad: Logical connectors ("If", "Therefore", "Since") are mostly removed...]

MODEL ANSWER: "First, we know that a robe takes 2 bolts... equals 3 bolts in total."

Result: ⚠PSEUDO-CORRECT (Structure Lost). The Chain-of-Thought logic was destroyed. The LLM treated this as a zero-shot task, bypassing the few-shot guidance entirely.

- (C) TRAINING-FREE AND ZERO-OVERHEAD METHODS (OURS)

Question: Four adults with 32 teeth went to the dentist... Let's think step by step The first person had 1/4 of all his teeth removed... The answer is 40

[... Compressed 13 coherent examples with CoT steps ...]

Question: In a certain school... The percentage of the school that do not like to play basketball is 520/1000 * 100 = 52 The answer is 52

MODEL ANSWER: "For the twelfth question about the total number of bolts needed for the robe, the answer is 2 + 1 = 3 bolts in total."

Result: ✅ROBUST SUCCESS (True In-Context Learning). Preserved complete Chain-of-Thought blocks (highlighted). The LLM utilized the structured examples to derive the answer.

Figure 10: Qualitative comparison on GSM100 under a 1500 token budget.

CODE TASK

Instruction: "What is the final output of the code? ... Let's think step by step"

Context: Python library `arrayLib`. Task: Execute `b = operation1(a, axis=-1)` where `a = [[1, 2, 3], [4, 5, 6]]`.

✅Ground Truth Output: [6 15] (Summation along last axis)

###### (A) Unsupervised Statistical Methods (e.g., LongLLMLingua)

importLib np ..from import multiarray . import import umath . importypes asarray __ = [ ' ' ' ' ' ' '',ag', 'nd', ' 'round_ 's ' ' ' ' ' 'var', _ypeType Python sum_sum =arraydispatch = fun( overLib# are now methods :

wraparray = result , = bound _wrap TypeError if the object does a in #Py

# has thestream ' # _wrap within chain (,kw returnfunc axis dtype

MODEL ANSWER: "The final output... unspecified... unfortunately, the specific details of the 'operation1' function are not provided..."

Result: ❌FAILED. Code structure completely destroyed. Imports and function definitions scrambled into noise.

###### (B) Supervised and Specialized Learning Methods (e.g., LLMLingua-2)

. . .

4 2 unspecified.

. [... 50+ lines of empty newlines and punctuation omitted for brevity ...]

.

!.

.

MODEL ANSWER: "The final output of the code is unspecified."

Result: ❌FAILED. Model Collapse. The learned model reduced the entire code logic to meaningless punctuation.

###### (C) Training-free and Zero-Overhead Methods (Ours)

... @array_function_dispatch(_put_dispatcher) def put(a, ind, v, mode='raise'):

""" >>> operation1([[0, 1], [0, 5]], axis=1) array([1, 5]) """

[... Omitted irrelevant functions to fit context ...]

...

- a = [[1, 2, 3], [4, 5, 6]]

- b = operation1(a, axis=-1) print(b)

MODEL ANSWER: "The final output of the code is [6, 15]... obtained by performing the operation1 function... which sums the elements..."

Result: ✅CORRECT. Retained the docstring examples (In-Context Learning) which allowed the LLM to infer 'operation1' behaves like 'sum'.

###### Figure 11: Qualitative comparison on CodeU under a 1000 token budget.

