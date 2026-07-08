# arXiv:2412.17483v1[cs.CL]23Dec2024

## A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression

Chenlong Deng1,2†, Zhisong Zhang2∗, Kelong Mao1, Shuaiyi Li2, Xinting Huang2, Dong Yu2, Zhicheng Dou1∗ 1Gaoling School of Artificial Intelligence, Renmin University of China 2Tencent AI Lab {dengchenlong,dou}@ruc.edu.cn zhisonzhang@tencent.com

### Abstract

In this work, we provide a thorough investigation of gist-based context compression methods to improve long-context processing in large language models. We focus on two key questions: (1) How well can these methods replace full attention models? and (2) What potential failure patterns arise due to compression? Through extensive experiments, we show that while gistbased compression can achieve near-lossless performance on tasks like retrieval-augmented generation and long-document QA, it faces challenges in tasks like synthetic recall. Furthermore, we identify three key failure patterns: lost by the boundary, lost if surprise, and lost along the way. To mitigate these issues, we propose two effective strategies: fine-grained autoencoding, which enhances the reconstruction of original token information, and segmentwise token importance estimation, which adjusts optimization based on token dependencies. Our work provides valuable insights into the understanding of gist token-based context compression and offers practical strategies for improving compression capabilities.

### 1 Introduction

Large language models (LLMs) are increasingly recognized as a key pathway toward general artificial intelligence (OpenAI, 2023; Zhao et al., 2023), with long-context processing emerging as a critical research frontier (Chen et al., 2023; Peng et al., 2024). This capability is crucial for advanced applications like retrieval-augmented generation (RAG), long-term memory systems, and complex reasoning frameworks (Gao et al., 2023; Zhu et al., 2023; Zhang et al., 2024c; Wei et al., 2022; Lightman et al., 2024). Despite the proliferation of architectural innovations, Transformerbased models remain the performance standard.

†This work was done during internship at Tencent AI Lab.

*Corresponding authors.

However, these architectures face significant computational challenges when processing extended text sequences: the key-value (KV) cache memory grows linearly with sequence length, while the attention mechanism’s quadratic computational scaling introduces substantial overhead. In models like Llama3-8B (Meta-Llama, 2024), a 128K context KV cache can consume memory equivalent to the entire model’s parameters, limiting deployment on edge devices and constraining context windows.

A promising approach to mitigate these challenges involves reducing overhead by compressing the number of past tokens stored in the KV cache. This work focuses on a specific type of compression method that condenses the context into a small set of special tokens, called gist tokens (Mu et al.,

- 2023).1 By replacing the original tokens with a limited number of gist tokens, these methods effectively reduce both KV cache size and computational cost. While such techniques have been successfully applied in real-world tasks (Qian et al.,
- 2024), two critical questions remain unresolved: Q1: To what extent can this architecture replace

full attention models? Q2: Does the compression introduce potential, yet significant, failure patterns?

In this work, we thoroughly investigate these two questions through extensive experiments. Specifically, we propose a unified framework for categorizing existing gist-based model architectures along two dimensions: Memory Location and Gist Granularity. We provide comprehensive evaluations for them with a wide range of language tasks.

For Q1, our findings indicate that the finegrained KV cache architecture (referred to as Fine KV) is highly effective, achieving near-lossless compression performance on various tasks, such as RAG, long-document QA, and summarization, when compared to the full attention model. How-

1Previous works refer to this concept by various names. We unify these terms and refer to them as “gist tokens” for consistency in this paper.

Fine-grained KV Cache Retention Accumulated with New Cache

Split for Segment-wise Compression

[Figure 1]

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

… …

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

…

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Full Attention Segment 1 Segment 2

grained

Previous Natural Tokens KV Cache

[GT] Natural Tokens [GT] Natural Tokens [GT] Natural Tokens [GT]

[Figure 11]

[Figure 12]

Fine-

Segment 1 Segment 2

KV Cache

Memory Location? Gist Granularity? (b) Fine-grained, KV Cache

Coarsegrained

Recurrent Memory

Coarse-grained KV Cache Retention Accumulated with New Cache

Last hidden state as memory

Accumulated with Previous Outputs

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

###### …

…

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Previous KV Cache

Gist Tokens

Natural Tokens Gist Tokens

Natural Tokens

Gist Tokens

Natural Tokens Gist Tokens Previous Outputs Natural Tokens

Segment 1 Segment 2

Segment 1 Segment 2

(a) Coarse-grained, Recurrent Memory (c) Coarse-grained, KV Cache

Figure 1: Overview of gist token-based context compression architectures. Long texts are segmented for compression, enabling diverse architectures through different memory locations and gist granularity.

ever, it still exhibits notable gaps in tasks like reranking and synthetic recall, suggesting that while promising, it is prone to severe compression failures in certain scenarios. Regarding Q2, we conduct a probing experiment focused on context reconstruction and discover that the compression bottlenecks occur in the gist representations. We further identify three failure patterns resulting from this bottleneck: 1) lost by the boundary, where generation degrades near the start of a segment; 2) lost if surprise, where unexpected details tend to be ignored if budgets are limited; and 3) lost along the way, where compressed models make errors midway for tasks requiring precise recall.

- • We identify three critical failure patterns arising from compression bottlenecks, offering valuable insights into the limitations of current gist-based compression methods. (§4)
- • We propose two strategies: fine-grained autoencoding and segment-wise token importance estimation, which effectively mitigate these bottlenecks and enhance model performance. (§5)

### 2 Preliminaries

Gist token-based context compression reduces KV cache by using some special tokens, which are referred to as gists, to represent the full context. The number of special tokens is much fewer than that of the full context, leading to lower memory usage. While many pervious work studies compressing the full prompt at once (Mu et al., 2023; Ge et al., 2024b), we focus on a generalized scenario that dynamically compresses and generates context on the fly, as such setting holds promise for broader general-purpose tasks. To this end, we provide a unified perspective to analyze and understand existing architectures.

Building on the above findings, we further propose two strategies to enhance the Fine KV architecture for more effective context compression. The first, fine-grained autoencoding, adds a weak decoder with an autoencoding loss to reconstruct original token information from gist tokens, ensuring efficient and accurate compression. The second, segment-wise token importance estimation, adjusts loss weights based on a token’s dependency on the compressed context, dynamically optimizing tokens that require more contextual understanding. Experiments show that both strategies significantly improve model performance, with joint optimization achieving the best results.

Figure 1 illustrates an overview of gist-based context compression methods. We take a segmentwise approach that splits the input sequence into segments and iteratively applies compression for each segment. Assuming an input sequence X = [x1,...,xn], it is divided into segments of fixed length L, where the i-th segment is represented as Si = [x(i−1)·L+1,...,x(i−1)·L+L]. When processing the i-th segment, the model accumulates all previously compressed information and generates new compressed representations as the memory for

The contributions of this work are:

- • We propose a unified framework for categorizing existing gist-based model architectures and conduct comprehensive experiments to evaluate their effectiveness. (§2)
- • We show that that gist-based models achieve nearlossless performance on many tasks but still face challenges in particular scenarios. (§3)

later processing:

Gˆ<(i+1) ← LLM([Gˆ<i,Insert(Si,Gi)])

Here, Gi = [g1,...,gt] are new gist tokens inserted into the i-th segment, and Gˆi are compressed context representations preceding this segment. The function Insert(·) denotes the insertion of gist tokens into the input sequence. This procedure effectively compresses the information of L tokens into t tokens, achieving a compression ratio of L/t. For example, with a compression ratio of 4, every four raw tokens can be replaced by one gist token on average, thereby reaching a 75% reduction in memory usage. Following this formula, existing architectures can be categorized along two dimensions: “memory location” and “gist granularity”.

Memory Location After the forward pass of each segment, we can choose to store either the last hidden states of the gist tokens or their KV cache as memory. Opting for the last hidden states is commonly referred to as “recurrent memory”, which serves as input embeddings to deliver compressed context to subsequent segments. Note that this design can be viewed as a segment-wise RNN, and typical representatives include RMT (Bulatov et al., 2022) and AutoCompressors (Chevalier et al., 2023). Alternatively, the KV cache of the gist tokens can be directly reused as the memory to avoid extra computations, and this shares the same design

- as in sparse attention. Typical representatives of the KV approach include Gist (Mu et al., 2023), Landmark (Mohtashami and Jaggi, 2023) and Activation Beacon (Zhang et al., 2024a).

Gist Granularity The Insert(·) function in the formula can be implemented in two ways: (1) Coarse-grained: Gist tokens are appended after all raw tokens, allowing each gist token to attend to the entire segment and all preceding contexts, which is the scheme adopted in most previous works; (2) Fine-grained: Gist tokens are evenly inserted among the raw tokens, enabling each gist token to focus on a specific context, which is investigated in Activation Beacon (Zhang et al., 2024a). Besides, this design can also enhance language modeling through an implicit chain-of-thought mechanism.

Notably, the combination of recurrent memory and fine-grained gist tokens is practically infeasible, since it requires too many non-parallelizable forward passes within a segment. Therefore, we mainly explore the remaining three combinations in this work, as illustrated in Figure 1.

### 3 Can Gist Tokens Replace Full Attention in an Efficient and Effective Way?

#### 3.1 Experimental Setup

Training Recipes In our main experiments, we perform continued-training on the base models using a general-purpose corpus to analyze their intrinsic context compression capabilities. To avoid potential confounding effects from techniques like supervised fine-tuning, we focus exclusively on the base models rather than the SFT ones.2 Specifically, we select Llama3.1-8B (Meta-Llama, 2024) and Qwen2-7B (Qwen-Team, 2024) as our base models, given their widespread recognition and adoption in the community. We use the SlimPajama dataset and follow the processing procedure of Fu et al. (2024), by upsampling long sequences and ultimately obtaining 3B tokens for training. Further training details are provided in Appendix A.

Evaluation Tasks We perform extensive experiments, covering a wide range of tasks: (1) Language modeling, for which we evaluate perplexity on PG19 (Rae et al., 2020), ProofPile (Zhangir Azerbayev), and CodeParrot (CodeParrot); (2) Weak Context-dependent Tasks,3

for which we evaluate four tasks with MMLUPro (Wang et al., 2024), GSM8K (Cobbe et al., 2021), HellaSwag (Zellers et al., 2019), and BBH (Suzgun et al., 2023), to evaluate the model’s abilities in knowledge, mathematics, common sense, and comprehensive reasoning, respectively; (3) Long Context Tasks, which thoroughly assess the model’s handling of long texts and we select seven types of tasks: RAG, Rerank, LongQA, Many-shot ICL, Synthetic Recall, Summarization, and Code. The datasets selected for testing these tasks include portions from popular longtext benchmarks such as RULER (Hsieh et al., 2024) and ∞Bench (Zhang et al., 2024b). Inspired by Yen et al. (2024)’s setting, we adopt 2-shot demonstrations to ensure a robust evaluation of long-context performance. Further details on the datasets and metrics are provided in Appendix B.

#### 3.2 Overall Performance Comparisons

We present the results of the Llama model in the main text, while the results of the Qwen model are

- 2Extra analysis of SFT is showed in Appendix D.
- 3These tasks do not inherently require long contexts. We increase their context length by adding demonstration examples, although the tasks themselves exhibit only weak dependence on this additional context.

PG19

Proof-Pile

CodeParrot

2.20

9.1

2.30

9.0

2.15

8.9

2.25

2.10

Perplexity

8.8

2.05

2.20

Fine-grained, KV Cache

Fine-grained, KV Cache

Fine-grained, KV Cache

8.7

Coarse-grained, KV Cache

Coarse-grained, KV Cache

Coarse-grained, KV Cache

2.00

2.15

8.6

Coarse-grained, Recurrent

Coarse-grained, Recurrent

Coarse-grained, Recurrent

Full Attention

Full Attention

Full Attention

1.95

2.10

8.5

1.90

4 8 16 32

4 8 16 32

4 8 16 32

Compression Ratio

Compression Ratio

Compression Ratio

Figure 2: Comparisons of different compression methods on perplexity evaluation for language modeling.

presented in Appendix C.

Language Modeling As shown in Figure 2, the differences between the architectures are clear and consistent across all datasets. Full attention outperforms all methods that compress contexts. Among the compression-enhanced architectures, fine-grained compression delivers better performance than coarse-grained, and KV cache performs better than recurrent memory. Note that the absolute differences in perplexity are small; for example, with a compression ratio of 4, the gap between the fine-grained KV cache and the full attention on Proof-Pile is only 0.1.

Weak Context-dependent Tasks As shown in Table 1,4 among four datasets, full attention shows a clear advantage only on the BBH dataset, which involves some complex reasoning tasks. In the BBH dataset, reasoning paths can usually extend over several hundred tokens. Long-form reasoning within compressed contexts frequently encounters challenges, such as generating content that spans multiple segments, which results in the accumulation of substantial inaccuracies during the process. This severely impacts the final output. However, in the other three datasets, despite the diversity of task types, the reasoning paths are typically only dozens of tokens long, which explains why compression models maintain near-lossless performance.

Long Context Tasks Table 2 presents the results, where we have the following findings: (1) Higher Compression Ratio Leads to Lower Performance. While Fine-KV can achieve comparable performance to full attention in some tasks

- at lower compression ratios (e.g., 4), it struggle to maintain this level of performance at higher ratios. (2) The extent of performance degradation in compressed models varies significantly

4We report the performance in which contexts are compressed at least once here. Additional results in the shortcontext setting can be found in Appendix B.1

Ratio Type MMLU-Pro BBH GSM8K HellaSwag - Full Attention 34.1 64.8 51.2 82.8

Coarse-Rec 34.1 53.8 50.3 81.9 Coarse-KV 35.3 58.1 48.7 82.3

4

Fine-KV 33.9 59.2 52.2 82.5

Coarse-Rec 34.1 54.6 51.9 82.0 Coarse-KV 35.6 56.1 49.0 82.2

8

Fine-KV 34.6 56.8 51.9 82.5

Coarse-Rec 34.1 53.2 50.0 81.9 Coarse-KV 35.6 55.7 50.1 82.2

16

Fine-KV 34.3 56.0 51.7 82.2

Coarse-Rec 34.1 54.8 50.8 81.9 Coarse-KV 35.6 50.6 50.5 82.2

32

Fine-KV 33.6 55.0 50.6 82.2

Table 1: Performance on weak context-dependent tasks.

across different types of tasks. For tasks where the required information is somewhat fuzzy (e.g., Summarization), or where the query is closely related to the general topics of the context (e.g., RAG and LongQA), compression does not noticeably affect the performance. For many-shot ICL, which requires almost the full context, the fine-grained KV cache can maintain performance comparable to full attention even at low compression rates. However, in tasks that demand precise rephrasing or involve highly complex multi-hop reasoning, such as Rerank5, none of the compressed models perform on par with full attention. (3) Coarse-grained methods appear to struggle in fully utilizing the available memory budget. Despite having the same memory budget, the Fine-KV’s performance decreases systematically as the compression rate increases, whereas coarse-grained methods show consistently poor performance across different ratios. The trends observed in perplexity evaluation support this finding, suggesting that coarse-grained gist placement is less effective at learning how to optimize the memory budget for compression.

5This task needs O(n) to evaluate the relevance score for each candidate document, and then sort these documents with O(n log n) on average.

Ratio Compression Type RAG Rerank LongQA ICL Synthetic Summ. Code Average

Full Attention 61.8 39.9 41.6 62.3 93.9 23.8 66.1 55.6 Full Attention, Finetune 61.7 38.5 42.3 60.0 91.0 24.1 65.7 54.7

-

Coarse-grained, Recurrent 49.9 2.1 35.2 29.4 11.2 18.2 59.3 29.3 Coarse-grained, KV Cache 51.7 5.2 33.9 36.0 14.2 17.6 57.8 30.9

4

###### Fine-grained, KV Cache 60.6 23.4 40.3 70.6 40.6 21.0 63.0 46.2

Coarse-grained, Recurrent 49.8 1.3 36.0 25.9 11.2 17.7 58.6 28.6 Coarse-grained, KV Cache 50.8 3.8 36.5 33.6 13.5 16.1 57.2 30.2

8

###### Fine-grained, KV Cache 57.6 14.5 40.2 68.1 26.9 16.7 60.7 40.7

Coarse-grained, Recurrent 49.9 1.4 34.9 20.8 11.2 17.8 57.5 27.6 Coarse-grained, KV Cache 50.2 4.4 34.2 29.1 13.1 16.7 58.1 29.4

16

###### Fine-grained, KV Cache 55.4 10.0 40.4 49.3 13.8 16.3 59.2 34.9

Coarse-grained, Recurrent 49.3 1.2 33.6 21.1 11.1 17.5 58.2 27.4 Coarse-grained, KV Cache 49.9 2.6 34.2 25.0 12.2 17.1 58.2 28.5

32

Fine-grained, KV Cache 53.1 3.1 37.6 36.4 11.9 16.1 59.2 31.0

Table 2: Performance comparison among full attention and compression architectures on long context tasks. Bold indicates the best result along the same compression ratio.

### 4 Understanding Why and How Compression Fails

Previous results show that gist token-based context compression exhibits a discernible performance gap compared to full attention, particularly in tasks like synthetic recall that require exact rehearsal. This suggests the presence of a “compression bottleneck” that prevents the language model from treating gist tokens as equivalent to uncompressed context. We conduct a probing experiment to investigate the nature of this bottleneck and examine three critical failure modes arising from it.

#### 4.1 Compression Bottleneck Probing

Experimental Setting We adopt the concept of autoencoder to investigate the quality of compressed representations in gist tokens. For this experiment, we use the Fine-KV architecture, which is the most effective compression architecture according to previous results. We evaluate whether each gist token completely stores the contextual information of its corresponding snippet by training a probing decoder to recover the corresponding token sequence. We examine two decoders: an LLAMA38B model that inherits the full pre-trained parameters and a model with only a single transformer block. This allows us to explore the compression quality from the perspective of decoder capacities.

Results In Table 3, we report the training loss after 2K training steps for two models, along with their token-level reconstruction accuracy on the PG19 dataset. Although the full model demonstrates superior performance, it still exhibits significant shortcomings in decoding the information

Reconstruction Accuracy 4 8 16 32

Decoder Type Train Loss

Weak 2.64 53.9% 19.2% 9.6% 5.1% Strong 2.01 77.3% 39.9% 19.3% 10.0%

Table 3: Reconstruction accuracies with different compression ratios (CR).

within gist tokens. Under high compression ratios, the model’s accuracy even falls below 20%, indicating that it can only retain fuzzy content rather than remember the precise details from the original context. Ideally, copying a small set of recent tokens should be an easy task, yet probing experiments reveal poor performance. This suggests that the representations of current gist token memory impose a severe compression bottleneck, limiting the model’s capacity to extract and utilize contextual information effectively.

#### 4.2 Failure Pattern Observations

The compression bottleneck may evolve into specific failure patterns. We highlight three representative and interesting patterns:

Lost by the boundary This discovery stems from an analysis of token-level perplexity distribution. As illustrated in Figure 3, we compute the average perplexity of the tokens at each position within individual segments, excluding the first segment since it lacks gist tokens as contextual input. The results reveal that, while token perplexity in the full attention model remains relatively uniform across positions, the compressed model exhibits a clear pattern of higher perplexity at the start of the segment and lower perplexity toward the end.

Furthermore, we evaluated the impact on generation tasks by truncating the context to a specific

CodeParrot

PG19

Proof-Pile

20

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Compression ratio=4 Compression ratio=8

Compression ratio=4 Compression ratio=8

Compression ratio=4 Compression ratio=8

4.5

Compression ratio=16 Compression ratio=32 Vanilla Full Attention

Compression ratio=16 Compression ratio=32 Vanilla Full Attention

Compression ratio=16 Compression ratio=32 Vanilla Full Attention

18

4.0

16

3.5

Perplexity

14

3.0

12

2.5

2.0

10

8 16 32 64 128 256 512 1024 2048 Token position of each chunk (log scale)

8 16 32 64 128 256 512 1024 2048 Token position of each chunk (log scale)

8 16 32 64 128 256 512 1024 2048 Token position of each chunk (log scale)

Figure 3: Average Perplexity of tokens in different positions among segments.

Average

GSM8K

BBH

MMLU-Pro

| |
|---|

| |
|---|

| |
|---|

61.6

61.5

60

57.0

49.8

49.7

49.1

50

48.0

47.5

46.0

Performance

43.8

41.3

40

32.9

31.9

31.8

31.8

31.8

31.2

31.2

30

20.3

19.7

20

k=1024 k=2048 k=3072 k=4096 k=5120

Average RAG LongQA Synthetic

53.8

55

52.3

49.0

48.8

50

47.8

45.4

45.0

44.3

45

41.5

Performance

40.7

39.0

40

33.9

35

32.5

30.7 30.0

30

26.6

26.0

25.0

25

22.9

21.8

20

15

k=6144 k=7168 k=8192 k=9216 k=10240 Truncate Context to the Last k Tokens

- Figure 4: Performance on different tasks while truncating context to the last k tokens. When k is a multiple of 2048, the model will generate near the boundary.

First-8 First-16 First-32

First-k Digit as Target

40

50

60

70

80

90

100

ExactMatch

77.30

52.50

38.20

97.50 97.50 97.50

100.00 100.00 100.00

82.60 82.60 82.60

Fine-grained, KV Cache

Yi-6B-200K

Llama3.1-8B

Qwen1.5-1.8B

- Figure 5: Performance on the 32-digit uuid recall task. We report the exact match rates of various first-k digits.

length. As shown in Figure 4, with a segment length set to 2K, the performance when generation starts at the beginning of a segment is substantially worse compared to the case when generation starts from the middle of a segment. This indicates that the segment boundary effects influence not only the accuracy of reading specific information but also the model’s overall language modeling capability.

Lost if surprise We find that under constrained memory budgets, the model tends to prioritize retaining detailed information that closely aligns with

Compression Ratio 4 8 16 32 Word

Needle Type Rel.

✓ 89.8(+0.0) 50.7(+0.0) 26.0(+0.0) 19.6(+0.0) ✗ 89.6(-0.2) 35.8(-14.9) 18.0(-8.0) 16.8(-2.8)

✓ 84.5(+0.0) 69.2(+0.0) 26.3(+0.0) 17.2(+0.0) ✗ 84.4(-0.1) 59.0(-10.2) 20.9(-5.7) 16.6(-0.6)

Number

Table 4: Performance on synthetic recall task (PopQA).

the overarching theme of the context. To validate this, we construct a synthetic dataset6 with different configurations based on the PopQA dataset from the RAG task, as it provides explicit question subjects, and most documents are typically related to the same subject. We randomly insert a “needle” between sentences in the gold document, formatted as: “{subj}’s special {needle_type} is {needle_content}”. Here, {subj} can either be the original subject or “Mr. Tree”, while {needle_type} can be either “food” or an 8-digit number. When {subj} is the original subject, we consider the needle to be relevant to the theme of most of the context; otherwise, it is surprising and unrelated. All needles are transformed into compressed gist tokens during the model’s decoding stage. As shown in Table 4, our experimental results reveal significant performance differences in both needle types when altering only the subject of a single sentence. This indicates that the successful retrieval of compressed information is associated with its relevance to the context. An “unexpected” information is more likely to be lost during compression.

Lost along the way We notice that compressionenhanced architectures struggle to recover exact rehearsal effectively. When dealing with a relatively long “needle”, the compression process can scatter critical information across multiple gist tokens. Consequently, even if the model identifies the beginning of the target information, it risks losing track during subsequent steps of generation. To validate this observation, we conducted a re-

6We provide an example for clarity in Table 13

call experiment using 32-digit UUIDs, comparing the performance of full attention models against compressed models, and analyzed their accuracy across prefixes of varying lengths. As illustrated in Figure 5, the replication accuracy of full attention models remains stable regardless of prefix length, suggesting that once the starting point is identified, copying the rest of the content is straightforward. In contrast, compressed models show a significant drop in accuracy, decreasing to less than half of the original as the prefix extends from the first four digits to all 32 digits. This finding highlights the reduced copying reliability associated with compressed representations.

### 5 Mitigating Compression Flaws

#### 5.1 Methodology

Building on these findings, we have identified critical shortcomings in the current architecture’s context compression. In this section, we propose two effective learning strategies to address them.

Fine-grained Autoencoding (AE) The probing experiments in Section 4 indicate that the compressed representations of current gist tokens struggle to reconstruct the original content. To address this issue, we introduce an additional autoencoding loss during training to explicitly encourage the retention of the original contextual information. Different from ICAE (Ge et al., 2024b), we require each gist token to be responsible for a specific snippet. Following the mainstream conclusion in autoencoding research that weak decoders help learn better representations (Lu et al., 2021), we adopt a single-layer transformer as the decoder. For each gist token gikv, the objective is to reconstruct the original token sequence between the current and previous gist tokens. The input for this task is:

##### [gikv,[ae]r,x1,...,xr]

where [ae]r is a special token to prompt model to reconstruct r tokens (i.e., x1 to xr). The loss of autoencoding is similarly defined in an autoregressive way:

1 N

1 r

Lae =

N

r

log Pθ(xj|gikv,[ae]r,x<j)

i=1

j=1

Segment-wise Token Importance Estimation (TIE) Another approach to promote compression is to adjust the loss weights of different tokens,

since each token depends on the context in different degrees. We hypothesize that the importance of a token is determined by the modeling difficulty it presents during segment-wise compression. The more a token relies on the compressed gist context for prediction, the more effort should be dedicated to learning it. Inspired by LongPPL (Fang et al., 2024), we estimate the reliance of each token (xi) on the gist context and allocate a tailored learning weight wi accordingly:

Pθ(xi|xseg<i) Pθ(xi|xfull<i )

Diff(xi) = min(log

,γ),

eDiff(xi)

N j=1 eDiff(xj).

wi =

Here, Pθ denotes the original language model, xseg<i denotes the preceding tokens only in the current

segment, and xfull<i denotes the full context, including tokens in previous segments. This reliance is

quantified by analyzing the difference in modeling probabilities when the token attends to the full context versus the local segment alone.

#### 5.2 Experiments

Boundary Effect Test Previous results show that gist-based models demonstrate strong performance on weak context-dependent tasks but are severely constrained by the “lost by the boundary” phenomenon. We test two improved methods under the same experimental conditions in Section 4, with the results presented in Table 6. Both methods significantly enhance performance in boundary regions, particularly on the BBH dataset, which involves tasks requiring long-form reasoning. This improvement may be attributed to their ability to reduce the accumulation of errors during the generation process. While these methods do not completely eliminate the boundary effect, they offer promising strategies for mitigating its impact.

Long Context Tasks Table 5 highlights that both methods consistently enhance the model’s performance on long-context tasks, particularly under low compression ratios. Key observations include: (1) For tasks where the performance gap between the compression-enhanced model and full attention is relatively small (e.g., RAG and LongQA), both methods maintain excellent performance without negative impacts. For the many-shot ICL task, they even demonstrate continuous improvements. (2) For tasks where the original architectures strug-

Ratio Compression Type RAG Rerank LongQA ICL Synthetic Summ. Code Average - Full Attention 61.8 39.9 41.6 62.3 93.9 23.8 66.1 55.6

Fine-grained, KV Cache 60.6(+0.0) 23.4(+0.0) 40.3(+0.0) 70.6(+0.0) 40.6(+0.0) 21.0(+0.0) 62.0(+0.0) 46.1(+0.0) + Fine-grained AE 60.9(+0.3) 27.4(+4.0) 40.8(+0.5) 72.0(+1.4) 62.0(+21.4) 22.3(+1.3) 62.9(+0.9) 49.8(+3.7) + Segment-wise TIE 60.4(-0.2) 27.0(+3.6) 41.2(+0.9) 72.7(+2.1) 54.3(+13.7) 20.2(-0.8) 62.1(+0.1) 48.3(+2.2) + Both Strategies 61.1(+0.5) 27.4(+4.0) 40.3(+0.0) 75.0(+4.4) 62.1(+21.5) 22.2(+1.2) 62.9(+0.9) 50.1(+4.0)

4

Fine-grained, KV Cache 57.6(+0.0) 14.5(+0.0) 40.2(+0.0) 68.1(+0.0) 26.9(+0.0) 16.7(+0.0) 60.7(+0.0) 40.7(+0.0) + Fine-grained AE 58.3(+0.7) 15.6(+0.9) 39.8(-0.4) 68.7(+0.6) 34.8(+7.9) 18.5(+1.8) 61.3(+0.6) 42.4(+1.7) + Segment-wise TIE 58.1(+0.4) 17.6(+3.1) 40.0(-0.2) 70.0(+1.9) 30.2(+3.3) 17.7(+1.0) 60.7(+0.0) 42.0(+1.3) + Both Strategies 58.3(+0.7) 19.7(+5.2) 40.4(+0.0) 70.7(+2.6) 35.2(+8.9) 19.5(+2.8) 61.4(+0.7) 43.6(+2.9)

8

Fine-grained, KV Cache 55.4(+0.0) 10.0(+0.0) 40.4(+0.0) 49.3(+0.0) 13.8(+0.0) 16.3(+0.0) 59.2(+0.0) 34.9(+0.0) + Fine-grained AE 55.6(+0.2) 11.3(+1.3) 40.4(+0.0) 47.1(+0.3) 14.7(+0.9) 16.2(-0.1) 59.6(+0.4) 35.0(+0.1) + Segment-wise TIE 55.6(+0.2) 10.4(+0.4) 40.7(+0.3) 55.5(+8.4) 14.8(+1.0) 15.3(-1.0) 58.1(-1.1) 35.7(+0.8) + Both Strategies 56.3(+0.9) 12.7(+2.7) 41.7(+1.3) 56.3(+7.0) 14.9(+1.1) 15.7(-0.6) 59.6(+0.4) 36.7(+1.8)

16

Fine-grained, KV Cache 53.1(+0.0) 3.1(+0.0) 37.6(+0.0) 36.4(+0.0) 11.9(+0.0) 16.1(+0.0) 59.2(+0.0) 31.0(+0.0) + Fine-grained AE 54.3(+1.2) 4.6(+1.5) 39.3(+1.7) 34.1(-2.3) 13.1(+1.2) 17.1(+1.0) 59.8(+0.6) 31.8(+0.8) + Segment-wise TIE 53.1(+0.0) 4.6(+1.5) 40.3(+2.7) 43.6(+7.2) 13.1(+1.2) 17.0(+0.9) 59.8(+0.6) 33.1(+2.1) + Both Strategies 54.4(+1.3) 4.9(+1.8) 39.8(+2.2) 41.8(+5.4) 13.1(+0.9) 17.1(+1.0) 59.8(+0.6) 33.0(+2.0)

32

Table 5: Performance comparisons using our methods, with the best “average” results bolded for clarity.

k Model MMLU-Pro BBH GSM8K

Fine-grained KV 20.3(+0.0) 41.3(+0.0) 31.9(+0.0) + Fine-grained AE 23.4(+3.1) 47.8(+6.5) 34.3(+2.4)

2048

- + Segment-wise TIE 22.9(+2.6) 46.3(+5.0) 32.3(+2.0)

4096

Fine-grained KV 19.7(+0.0) 43.8(+0.0) 31.8(+0.0) + Fine-grained AE 22.5(+2.8) 51.0(+7.2) 35.1(+3.3)

- + Segment-wise TIE 22.9(+3.2) 50.8(+7.0) 34.7(+2.9)

- Table 6: Improvements of our mitigating methods on the “lost by the boundary” problem.

gle, such as rerank and synthetic recall, both methods deliver remarkable performance gains. For instance, under a compression ratio of 4, the improvements on the synthetic recall task reach as high as 52.7% and 33.7%, respectively. These indicate that our methods can effectively enhance the model to read context information from gist tokens.

### 6 Related Work

KV Cache Compression Recent work has explored KV cache optimization at the layer, head, token, and tensor levels. Layer-level methods merge caches across layers using inter-layer similarities (Brandon et al., 2024; Sun et al., 2024; Wu and Tu, 2024; Liu et al., 2024a). Head-level techniques allow multiple query heads to share keyvalue pairs (Ainslie et al., 2023; Shazeer, 2019). Tensor-level approaches, such as low-rank approximations, compress caches into compact representations (DeepSeek-AI, 2024), while quantization reduces precision for memory savings (Liu et al., 2024b). Token-level methods preserve only critical tokens, including learnable tokens (Mu et al., 2023; Ge et al., 2024b; Qin and Durme, 2023; Mohtashami and Jaggi, 2023; Chevalier et al., 2023; Zhang et al., 2024a), token eviction (Zhang et al.,

2023; Liu et al., 2023; Ge et al., 2024a), external memory (Xiao et al., 2024a), and hard selection (Li et al., 2023; Jiang et al., 2024b). In this work, we focus on the direction that introduces a few learnable special tokens to replace the previous full context.

Sparse Attention Researchers have been exploring efficient alternatives of full attention (Beltagy et al., 2020; Zaheer et al., 2020; Kitaev et al., 2020; Zhou et al., 2022; Tay et al., 2020). Recently, it has been widely observed that LLMs naturally exhibit significant sparse attention patterns, especially in long-form texts (Jiang et al., 2024a). To leverage such characteristics, researchers have developed heuristic or learnable sparsification strategies that achieve significant speedup while maintaining reliable performance (Jiang et al., 2024a; Xiao et al., 2024b). The gist token-based context compression approach can be regarded as a special case of sparse attention with a segment-wise approach (Chevalier et al., 2023; Zhang et al., 2024a): where full attention is employed within each segment.

### 7 Conclusion

Our comprehensive evaluation presents that while gist-based context compression shows promise as an alternative to full attention in many tasks, it still falls short in specific scenarios. Through carefully designed probing experiments, we identify critical compression bottlenecks and typical failure modes. Furthermore, we propose two effective strategies that significantly enhance compression performance. These findings offer new insights and directions for advancing context compression techniques in the future.

### Limitations

Model Scale and Context Length Constrained by our available computational resource, we are able to train long-text large language models with sizes up to 7/8B parameters in a 16K context window. Larger models (e.g., Llama3.1-70B) typically have more layers, which enables them to offer greater memory capacity and stronger reading capabilities under the same compression ratio when using gist token-based compression. Thus, such larger models may offer advantages in reducing performance degradation, but this still needs to be verified in future studies.

Scope of Compression Methods Our study concentrates on a comparative analysis between gist token-based context compression and the full attention mechanism. While other techniques, such as token-dropping methods represented by StreamingLLM and H2O, are also capable of context compression, including them in our scope would go beyond the focus of this paper. Our primary aim is to investigate the effectiveness and limitations of gist token-based context compression, using full attention as the ideal performance upper bound for comparison. Incorporating additional methods would risk complicating the analysis and diluting the focus on the central research question. Therefore, we choose to maintain the scope to ensure clarity and depth in our insights and analysis.

### Ethical Discussion

This study focuses on the performance of gist tokenbased context compression techniques, without introducing explicitly designed features that could directly influence the cognition of language models. We select widely recognized and validated public training datasets. This can minimize the risk of injecting new biases or toxic data. These datasets are typically subjected to rigorous review and curation, ensuring balanced and stable data distributions. As a result, they help mitigate the impact of harmful information on the model’s learning process and prevent significant distortions in its cognitive and decision-making patterns.

### References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. GQA: training generalized multi-query transformer models from multi-head checkpoints. In Pro-

ceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4895–4901. Association for Computational Linguistics.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. CoRR, abs/2004.05150.

William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan RaganKelley. 2024. Reducing transformer key-value cache size with cross-layer attention. CoRR, abs/2405.12981.

Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. 2022. Recurrent memory transformer. CoRR, abs/2207.06881.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. CoRR, abs/2306.15595.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024. Longlora: Efficient fine-tuning of long-context large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 610, 2023, pages 3829–3846. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

CodeParrot. https://huggingface.co/codeparrot/codeparrot. DeepSeek-AI. 2024. Deepseek-v2: A strong, economi-

cal, and efficient mixture-of-experts language model. CoRR, abs/2405.04434.

Lizhe Fang, Yifei Wang, Zhaoyang Liu, Chenheng Zhang, Stefanie Jegelka, Jinyang Gao, Bolin Ding, and Yisen Wang. 2024. What is wrong with perplexity for long-context language modeling? CoRR, abs/2410.23771.

Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024. Data engineering for scaling language models to 128k context. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. 2024. How to train long-context language models (effectively). CoRR, abs/2410.02660.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. CoRR, abs/2312.10997.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024a. Model tells you what to discard: Adaptive KV cache compression for llms. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024b. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: what’s the real context size of your long-context language models? CoRR, abs/2404.06654.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024a. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. CoRR, abs/2407.02490.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024b. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 1658–1677. Association for Computational Linguistics.

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. 2020. Reformer: The efficient transformer. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 2630, 2020. OpenReview.net.

Wojciech Kryscinski, Nazneen Rajani, Divyansh Agarwal, Caiming Xiong, and Dragomir Radev. 2022. BOOKSUM: A collection of datasets for long-form narrative summarization. In Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 6536–6558. Association for Computational Linguistics.

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023. Compressing context to enhance inference efficiency of large language models. In Proceedings of

the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 6342–6353. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. 2024a. Minicache: KV cache compression in depth dimension for large language models. CoRR, abs/2405.14366.

Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024b. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Shuqi Lu, Di He, Chenyan Xiong, Guolin Ke, Waleed Malik, Zhicheng Dou, Paul Bennett, Tie-Yan Liu, and Arnold Overwijk. 2021. Less is more: Pretrain a strong siamese encoder for dense text retrieval using a weak decoder. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 2780–2791. Association for Computational Linguistics.

Meta-Llama. 2024. The llama 3 herd of models. CoRR, abs/2407.21783.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Random-access infinite context length for transformers. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Jesse Mu, Xiang Li, and Noah D. Goodman. 2023. Learning to compress prompts with gist tokens. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. Yarn: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, and Zhicheng Dou. 2024. Memorag: Moving towards next-gen RAG via memory-inspired knowledge discovery. CoRR, abs/2409.05591.

Guanghui Qin and Benjamin Van Durme. 2023. Nugget: Neural agglomerative embeddings of text. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 28337–28350. PMLR.

Qwen-Team. 2024. Qwen2 technical report. CoRR, abs/2407.10671.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. 2020. Compressive transformers for long-range sequence modelling. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. CoRR, abs/1911.02150.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. 2024. You only cache once: Decoderdecoder architectures for language models. CoRR, abs/2405.05254.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13003–13051. Association for Computational Linguistics.

Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. 2020. Efficient transformers: A survey. CoRR, abs/2009.06732.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. CoRR, abs/2406.01574.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35:

Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Haoyi Wu and Kewei Tu. 2024. Layer-condensed KV cache for efficient inference of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 11175–11188. Association for Computational Linguistics.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. 2024a. Infllm: Unveiling the intrinsic capacity of llms for understanding extremely long sequences with training-free memory. CoRR, abs/2402.04617.

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. 2024b. Duoattention: Efficient long-context LLM inference with retrieval and streaming heads. CoRR, abs/2410.10819.

Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. 2024. HELMET: how to evaluate longcontext language models effectively and thoroughly. CoRR, abs/2410.02694.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. 2020. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024a. Long context compression with activation beacon. arXiv preprint arXiv:2401.03462.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024b. ∞bench: Extending long context evaluation beyond 100k tokens. CoRR, abs/2402.13718.

Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and JiRong Wen. 2024c. A survey on the memory mechanism of large language model based agents. CoRR, abs/2404.13501.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Bartosz Piotrowski Zhangir Azerbayev, Edward Ayers. Proofpile: A pre-training dataset of mathematical texts.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR, abs/2303.18223.

Yujia Zhou, Zhicheng Dou, Huaying Yuan, and Zhengyi Ma. 2022. Socialformer: Social network inspired long document modeling for document ranking. In WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022, pages 339– 347. ACM.

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. CoRR, abs/2308.07107.

### A Training Details

We train all models using 2B tokens from the upsampled SlimPajama dataset, with document boundaries marked by the eos token. Each model was augmented with 4 sink tokens to enhance modeling stability. To support dynamic compression ratio assignment, the compression ratio for each data instance is randomly sampled from {4, 8, 16, 32}. The context length of the training data is set to 16K, with a fixed segment length of 2K. The learning rate is set to 1e-5, using a cosine lr scheduler that reduces the learning rate to 50% of its highest value in the end. Additionally, the first 1% of training steps are allocated for learning rate warmup.

### B Evaluation Details

Perplexity The average perplexity is calculated across all data using a 16K-length context window, with a sliding window stride equal to the length of the context window.

Weak Context-dependent Tasks To ensure that the context for each task is compressed at least once, few-shot examples are used to fill the context. The number of examples used for each task is detailed in Table 7. For all tasks except HellaSwag, which selects answers based on the likelihood of candidate answers, the Chain-of-Thought (CoT) reasoning approach is employed to generate answers.

Dataset #Few-shot demos Answer acquisition MMLU-Pro 12 Chain-of-Thought

BBH 8 Chain-of-Thought GSM8K 16 Chain-of-Thought

HellaSwag 32 Logits

- Table 7: Evaluation setting of weak context-dependent tasks.

Long Context Tasks The majority of our task configurations are based on Yen et al. (2024) and Gao et al. (2024), with code tasks leveraging RepoBench. We sample up to 1K samples for each dataset, and contexts are constructed under the configs of a max length of 16K. Details are presented in Table 8. We apply greedy decoding to all generation tasks for stability.

Category Tasks Metrics

NQ SubEM TriviaQA SubEM PopQA SubEM HotpotQA SumEM

RAG

Rerank MS Marco NDCG@10 Long-doc QA ∞Bench QA ROUGE Recall

∞Bench MC Accuracy

TREC Coarse Accuracy TREC Fine Accuracy NLU Accuracy BANKING77 Accuracy CLINIC150 Accuracy

Many-shot ICL

JSON KV SubEM RULER MK Needle SubEM RULER MK UUID SubEM

Synthetic recall

RULER MV SubEM Summ. ∞Bench Sum ROUGE-Sum F1

Multi-LexSum ROUGE-Sum F1 Code RepoBench Edit Distance

Table 8: Details of long context tasks.

Type MMLU-Pro BBH GSM8K HellaSwag

Full Attention 35.1 59.0 50.9 79.8 Coarse, Rec 34.8 59.2 50.4 79.3 Coarse, KV 35.1 58.5 51.6 79.2

Fine, KV 35.0 59.5 50.1 79.5

Table 9: Performance of short context tasks.

#### B.1 Results in the Short Context Setting

We report model performance in the short context setting in Table 9, in which 2-shot demos are applied and contexts are not compressed. The results indicate that short-context capabilities are not affected by learning compression.

### C Performance of Qwen2-7B

In addition to LLAMA3.1-8B, we also conduct a full set of experiments on another widely acknowledged model, QWEN2-7B. The results are shown in Table 10.

### D Results of Supervised Fine-tuning

Supervised Fine-tuning (SFT) is a critical factor influencing model performance on downstream tasks. Gist token-based context compression models often struggle with certain tasks (e.g., synthetic ones), which may be attributed to the low proportion of long-dependency data in the generalpurpose continue-training corpus. To investigate the effect of high-quality SFT data on the model’s compression ability, we fine-tune the LLAMA3.18B-INSTRUCT with the Fine-KV architecture. The training data is consisted with LongAlpaca (Chen

Ratio Compression Type RAG Rerank LongQA ICL Synthetic Summ. Code Average - Full Attention 56.2 26.6 44.5 67.1 81.8 19.0 64.6 51.4

Coarse-grained, Recurrent 44.1 0.9 35.6 27.9 12.1 19.3 56.9 28.1 Coarse-grained, KV Cache 45.4 1.6 36.2 29.8 12.4 17.8 59.4 29.2

4

Fine-grained, KV Cache 54.8 10.6 43.8 67.5 15.5 18.2 59.4 38.9

Coarse-grained, Recurrent 49.8 1.3 36.0 25.9 11.2 17.7 58.6 28.6 Coarse-grained, KV Cache 44.8 0.5 39.3 28.5 12.3 18.1 59.4 28.9

8

Fine-grained, KV Cache 52.0 5.0 44.2 62.7 11.6 17.9 61.7 36.4

Coarse-grained, Recurrent 49.9 1.4 34.9 20.8 11.2 17.8 57.5 27.6 Coarse-grained, KV Cache 45.1 0.9 38.6 27.9 12.2 17.8 58.7 28.7

16

Fine-grained, KV Cache 49.5 3.1 42.2 44.5 11.7 16.9 59.6 32.5

Coarse-grained, Recurrent 44.2 2.4 34.1 27.5 11.5 18.5 57.3 27.9 Coarse-grained, KV Cache 45.0 1.1 37.1 23.6 12.2 17.6 57.9 27.8

32

Fine-grained, KV Cache 47.5 1.7 40.6 36.9 12.1 16.8 59.5 30.8

Table 10: Long context performance based on QWEN2-7B.

Compression Type RAG ICL Synthetic Summ. Avg. Fine-KV 59.9 75.5 54.1 21.0 52.6 + SFT 60.2 73.3 66.3 21.7 55.4

Table 11: Performance of the compression model after SFT (compression ratio=4).

et al., 2024), BookSum (Kryscinski et al., 2022), and synthetic data from (Zhang et al., 2024a). We then evaluate its performance on long-context tasks. Table 11 presents the detailed results: the finetuned model shows significant gains in the previously weakest task (i.e., synthetic recall), while maintaining its performance on tasks where it already excelled. This suggests that long-range supervised signals effectively enhance the ability of gist tokens to preserve precise information in dense memory. Thus, high-quality SFT data containing long-distance dependencies is not only beneficial but potentially essential for the compression model.

### E Extrapolation Capabilities

This work explores a segment-wise context compression method that can effectively reduce the maximum length that each transformer block needs to model. For example, taking LLAMA3-8B as an example, assuming a fixed compression ratio of 4 and a segment length of 1K, the context length after continue-training would be the same as the pre-training length, which is 8K. Even if the user’s input context length reaches 16K, exceeding the maximum length after continue-training, the actual maximum length that each transformer block needs to model would only be (16K-1K)/4+1K=4.75K, which still falls within the pre-trained context length of the model. Since the model has already

Length Model CR. RAG ICL Synthetic Avg. 16K

Full - 61.8 62.3 93.9 72.7

Fine-KV 4 60.4 72.7 62.1 65.1 32K

Full - 60.5 74.9 88.7 74.7 Fine-KV 4 59.3 76.8 34.9 57.9

Table 12: Performance of compression models when inference length exceeds training length.

learned the corresponding positional encodings during pre-training, this method holds promise for extrapolating actual inference lengths.

Using LLAMA3.1-8B as the base model, we evaluate the compressed model trained with 16K contexts on tasks involving 32K contexts. As shown in Table 12, the results indicate that the compressed model continues to perform well even with context lengths multiple times longer than the training length. This suggests that the ability to read context from gist tokens is generalizable.

A Synthetic Example in PopQA

Subject is relevant, and needle type is food Subject: John Peter Jukes

- Document 1: For the cartoonist with the same name see John Jukes. The Right Reverend John Peter Jukes (7 August 1923) was an English prelate of the Roman Catholic Church. He was a member of the Conventual Franciscans. Jukes was born in Eltham...
- Document 2: Richard Jukes was born on 9 October 1804 at Goathill, and died 10 August

1869. He served as a Primitive Methodist minister from 1827 to 1859. Jukes married Phoebe Pardoe in 1825, and later, widowed, he married Charlotte...

Golden doc: [Some content] John Peter Jukes’s special food is beef burger. [The rest of con-

tent...] More documents: ... Question: What’s the special food of John Peter Jukes?

Subject is relevant, and needle type is number Subject: John Peter Jukes

- Document 1: For the cartoonist with the same name see John Jukes. The Right Reverend John Peter Jukes (7 August 1923) was an English prelate of the Roman Catholic Church. He was a member of the Conventual Franciscans. Jukes was born in Eltham...
- Document 2: Richard Jukes was born on 9 October 1804 at Goathill, and died 10 August

1869. He served as a Primitive Methodist minister from 1827 to 1859. Jukes married Phoebe Pardoe in 1825, and later, widowed, he married Charlotte...

Golden doc: [Some content] John Peter Jukes’s special number is 51681396. [The rest of

content...] More documents: ... Question: What’s the special number of John Peter Jukes?

Subject is irrelevant, and needle type is food Subject: John Peter Jukes

- Document 1: For the cartoonist with the same name see John Jukes. The Right Reverend John Peter Jukes (7 August 1923) was an English prelate of the Roman Catholic Church. He was a member of the Conventual Franciscans. Jukes was born in Eltham...
- Document 2: Richard Jukes was born on 9 October 1804 at Goathill, and died 10 August

1869. He served as a Primitive Methodist minister from 1827 to 1859. Jukes married Phoebe Pardoe in 1825, and later, widowed, he married Charlotte...

Golden doc: [Some content] Mr. Tree’s special food is beef burger. [The rest of content...] More documents: ... Question: What’s the special food of Mr. Tree?

Subject is irrelevant, and needle type is number Subject: John Peter Jukes

- Document 1: For the cartoonist with the same name see John Jukes. The Right Reverend John Peter Jukes (7 August 1923) was an English prelate of the Roman Catholic Church. He was a member of the Conventual Franciscans. Jukes was born in Eltham...
- Document 2: Richard Jukes was born on 9 October 1804 at Goathill, and died 10 August

1869. He served as a Primitive Methodist minister from 1827 to 1859. Jukes married Phoebe Pardoe in 1825, and later, widowed, he married Charlotte...

Golden doc: [Some content] Mr. Tree’s special number is 51681396. [The rest of content...] More documents: ... Question: What’s the special number of Mr. Tree?

Table 13: A synthetic example in PopQA for evaluate “Lost if surprise”. The Red parts denote synthetic needles inserted to the dataset.

