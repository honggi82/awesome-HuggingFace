## Trainable Dynamic Mask Sparse Attention

Jingze Shi13, Yifan Wu13, Yiran Peng13, Bingheng Wu13, Liangdong Wang2, Guang Liu2, Yuyu Luo∗1 1HKUST(GZ), 2BAAI, 3SmallDoges

# arXiv:2508.02124v6[cs.AI]16Nov2025

Abstract

The increasing demand for long-context modeling in large language models (LLMs) is bottlenecked by the quadratic complexity of the standard self-attention mechanism. The community has proposed sparse attention to mitigate this issue. However, position-aware sparse attention methods rely on static sparse structures that lack adaptability to diverse query contexts, while content-aware sparse attention methods depend on heuristic key-value selection, hindering full differentiability. We introduce a trainable dynamic mask sparse attention mechanism, a method that merges the advantages of both position-aware and content-aware approaches. Dynamic Mask Attention (DMA) achieves this through three key innovations: First, it leverages value vector representations to generate content-aware dynamic masks, enabling the model to adaptively identify and attend to critical information. Second, it computes position-aware sparse weights in a hardwarefriendly manner, efficiently skipping unnecessary computational regions. Finally, we demonstrate that the introduced dynamic mask and sparse weights do not obstruct gradients, supporting end-to-end training. We have validated the performance of DMA through comprehensive experiments. A large body of experimental evidence shows that DMA consistently holds a Pareto advantage over state-of-the-art sparse attention baselines in tasks including scaling laws, multiquery associative recall, standard benchmarks, and needle in a haystack tests, while also delivering up to a 10x overall speedup. These results highlight its ability to effectively balance model efficiency with long-context modeling capabilities. Our computational kernel code is now open-source at https://github.com/flash-algo/flash-sparse-attention to encourage further research and application by the community.

### 1 Introduction

Recent breakthroughs in large language models (LLMs) have yielded remarkable achievements in tasks requiring longcontext reasoning (Snell et al. 2024), such as deep reasoning (HuggingFace 2025), codebase generation (K. Zhang et al.

- 2024), and multi-turn autonomous agents (Park et al. 2023). A key factor underpinning these successes is the ability to effectively model long-range dependencies, often spanning thousands of tokens (DeepMind 2025; Guo et al. 2025; Team
- 2025). However, the standard self-attention mechanism (Vaswani et al. 2017) employed in Transformer architectures inherently exhibits quadratic computational complexity (Zaheer et al. 2020), which severely restricts scalability to longer sequences. Consequently, designing attention mechanisms that enhance computational efficiency without sacrificing modeling accuracy has become a critical research frontier for advancing the capabilities of LLMs.

Limitations of Existing Methods. Current efficient attention strategies primarily leverage two types of sparsity: the position sparsity (Martins and Astudillo 2016), which facilitates the efficient computation of essential query-key pairs, and the content sparsity (Ge et al. 2023), which enables the selective computation of relevant tokens. The first category includes methods such as sliding window attention (Beltagy, Peters, and Cohan 2020), which employs static structures; multi-head latent attention (A. Liu et al. 2024), which uses low-rank approximations; and native sparse attention (Yuan et al. 2025), which utilizes learnable compression weights. Although these approaches can achieve efficient long-context modeling, they often struggle to maintain their effectiveness. The second category encompasses KV cache eviction methods (Y. Li et al. 2024; Zhenyu Zhang et al. 2023; Z. Zhou et al. 2024); block-wise KV cache selection strategies that dynamically choose cache blocks based on relevance predictions (Y. Gao et al. 2024; Jiaming Tang et al. 2024; C. Xiao et al. 2024); and filtering methods that employ sampling (Z. Chen et al. 2024), clustering (G. Liu et al. 2024), or hashing (Desai et al. 2024). Despite their conceptual appeal, these techniques often fail to realize their theoretical speedups in practical deployments due to the overhead from dynamic computations or inaccurate sparsification decisions.

∗Corresponding author: Yuyu Luo (E-mail: yuyuluo@hkust-gz.edu.cn).

· ·

·

·

·

·

·

·

·

·

·

[Figure 1]

sm(QK??M) nh x n x w

Sparse Weights tion

For war d Backwar d

| | |Q nhx n x d|
|---|---|---|
| | | |

|h| |Computation| |
|---|---|---|---|
| | | |Skip calculation of|

| | | | |
|---|---|---|---|
| | | | |

@

Backward

| | |
|---|---|
| | |

Outputs

@

| |K nhx dhx n| |
|---|---|---|
| | | |

Ignored Token

Inputs

O nh x n x dh

A nh

M nh x n

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

f Top

Weights as Mask grad

V n x dh

| |nhx| |
|---|---|---|
| | | |

? nhx dhx nh

? nh x n

use TopW only in forward

Dynamic Mask Generation

Linear Projection

Lear nable Oper ation Par ams

Activated Token

Ignor ed Token

Figure 1: Workflow and Performance of Dynamic Mask Attention. Left: Overall workflow of DMA. The first step projects the input into 𝑄, 𝐾, and 𝑉. The second step generates content-aware dynamic masks. The third step computes sparse weights. Black solid arrows indicate the forward computation path, while gray dashed arrows represent the backward computation path. Right: Relative performance comparison between full attention and DMA on benchmark tests. DMA achieves higher recall rates and significantly faster speeds while maintaining competitive accuracy.

Table 1: Comparison of Different Attention Variants. Comparison of different Self-Attention mechanisms. 𝑛 denotes sequence length, 𝑑ℎ represents head dimension, 𝑤 is window size, 𝑑𝑐 is compressed dimension, 𝐵 is compression block size, and 𝑘 is selection budget. Complexities focus on attention weight computation and memory requirements. Trainable indicates whether the sparsity pattern can be learned end-to-end.

Mechanism Comp. Complexity Mem. Complexity Sparsity Trainable

MHA 𝑂(𝑛2𝑑ℎ) 𝑂(𝑛2) Static ✗ SWA 𝑂(𝑛𝑤𝑑ℎ) 𝑂(𝑛𝑤) Position-aware ✗ MLA 𝑂(𝑛2𝑑𝑐) 𝑂(𝑛2) Low-rank Approx ✓ NSA 𝑂(𝑛2𝑑𝑐/𝐵 + 𝑛𝑘𝐵𝑑ℎ + 𝑛𝑤𝑑ℎ) 𝑂(𝑛2/𝐵 + 𝑛𝑘𝐵) Hybrid ✓ H2O 𝑂(𝑛𝑘𝑑ℎ) 𝑂(𝑛𝑘) Content-aware ✗ InfLLM 𝑂(𝑛𝑘𝑑ℎ) 𝑂(𝑛𝑘) Content-aware ✗ Quest 𝑂(𝑛𝑘𝑑ℎ) 𝑂(𝑛𝑘) Content-aware ✗ DAM 𝑂(𝑛𝑘𝑑ℎ) 𝑂(𝑛𝑘) Content-aware ✗

DMA 𝑂(𝑛𝑤𝑑ℎ) 𝑂(𝑛𝑤) Content-Position Dual-aware ✓

Key Challenges. To overcome these limitations, an ideal sparse attention mechanism must simultaneously address two fundamental challenges: leveraging position-aware sparsity for essential computations (Child et al. 2019) and exploiting content-aware sparsity for selective computation (Z. Dai et al. 2019). Meeting both requirements is crucial for achieving efficient and effective long-context reasoning and training in practice. However, existing methods still exhibit limitations, often facing a trade-off between efficiency and effectiveness. This dilemma highlights the urgent need for attention mechanisms that can preserve information integrity while achieving computational efficiency.

Our Method. In order to address these challenges and achieve efficient and effective sparse attention mechanisms, we ingeniously integrate the strengths of both strategies, attempting to strike a balance between the two, and propose Dynamic Mask Attention (DMA) to tackle the challenges of long-context modeling. As shown in Table 1, compared to other attention variants, DMA is a trainable content-position dual-aware sparse attention mechanism. As illustrated in Figure 1, it leverages two core innovations: generating dynamic masks using content-aware sparsity and computing sparse weights using position-aware sparsity, allowing the model to focus on relevant tokens while ignoring irrelevant ones. Furthermore, all computational operations are designed to be continuously differentiable, enabling end-to-end training of dynamic mask attention via gradient descent.

2

·

+

InnerLoop

CausalBroadcastonHBM

InnerLoop

QK? nh

w

Outer Loop

Outer Loop

K? nh x dhx n

n x

###### M nh x n x n

x

Concatenate on HBM

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|

Q nh x n x dh

V nh x n x dh

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
|SRAM| | | | | | | |
| | | | | | | | |
| | | |Co|mput|e|on SR|AM|
| | | | | | | | |
| | | | | | | | |

Copy to SRAM

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | |Co|m|pute|on|SRAM|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | |Output to HBM| | | | |
| | | | | | | | |

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

Copy to

? nhx n

V nhx n x dh

sm(QK??M)V nh x n x dh

| | | | | | | |
|---|---|---|---|---|---|---|

| |
|---|

? nh x dhx nh

A nh

Outer Loop

Inner Loop

- Figure 2: Dynamic Mask Attention Architecture. Left: Content-Aware Mask Computation. The mask computation part of dynamic mask attention. In the outer loop, the stride weight Δ and gate weight 𝐴 are loaded into high-speed SRAM, and in the inner loop, the zero-order hold method is used to loop through the 𝑉 blocks loaded into SRAM, sampling from it to generate content-aware 𝐾 masks. These masks are then causally broadcast to the length of 𝑄 in HBM. Finally, in the outer loop, all mask blocks are concatenated to form the final content-aware sparse dynamic mask. Right: Position-Aware Weights Computation. The weight computation part of dynamic mask attention, where in the outer loop, the 𝐾 and 𝑉 blocks are looped and loaded into SRAM, and in the inner loop, the 𝑄 blocks are accessed, loaded into SRAM, and the output of the attention weight computation is written back to HBM. If the current position of the 𝐾 block is designated as masked in the dynamic mask, the attention weight at that position is directly filled with 0, skipping the computation at that position, forming the final position-aware sparse attention weights.

nh M

x

###### n

x

n

Kernel Design. We implement a dedicated CUDA kernel that merges the memory efficiency of FlashAttention (Dao et al. 2022) with DMA’s trainable sparsity, as illustrated in Figure 2, enabling hardware-level skipping of masked regions without incurring additional redundant computations. The kernel natively supports attention masks and biases with batch, head, and query broadcasting, ensuring flexible integration with diverse Transformer architectures. A block-level reduction determines the skip logic: tiles corresponding to all-zero masks bypass both computation and memory access, reducing the effective complexity from O(𝑛2) to O(𝑛·𝑤) for𝑤 ≪ 𝑛. The forward and backward passes share a unified skip logic, fetching 𝐾/𝑉 tiles only when necessary, thereby maintaining an O(𝑛) memory footprint without materializing the full attention matrix. The backward pass incorporates a complete gradient chain with fused bias gradients, rendering the entire pipeline fully differentiable for end-to-end training. To maximize throughput, we employ shared memory aliasing, pipelined prefetching, and coalesced memory accesses to minimize bandwidth pressure and improve hardware occupancy. These design choices allow DMA to sustain high performance on extremely long contexts, such as 128K+ tokens, while preserving accuracy comparable to dense attention baselines.

InnerLoop

Contributions. We make the following contributions: (i) We introduce Dynamic Mask Attention (DMA), a trainable, content- and position-aware sparse attention mechanism that decouples content-driven dynamic mask generation from position-aware sparse weight computation, reducing the effective time complexity from O(𝑛2) to O(𝑛·𝑤) and the memory footprint to O(𝑛·𝑤) for window size𝑤 ≪ 𝑛. (ii) We develop a fully differentiable CUDA kernel that fuses FlashAttentionstyle tiling with hardware-efficient mask skipping. The kernel supports batch/head/query broadcasting for masks and biases, employs block-level skip logic to avoid unnecessary computation and memory traffic, and maintains an O(𝑛) memory footprint without materializing the full attention weight matrix. (iii) We validate DMA across diverse application scenarios, showing consistent speedups over other optimized implementations, improved scaling laws (Hoffmann et al. 2022), higher accuracy on multi-query associative recall (Arora et al. 2024), competitive downstream benchmarks performance, and robust needle-in-a-haystack (Kamradt 2023) results.

OuterLoop

### 2 Rethinking Sparse Attention

Since the advent of the Transformer (Vaswani et al. 2017), the attention mechanism has become central to sequence modeling, yet its 𝑂(𝑛2) computational and memory complexity remains a bottleneck for processing long sequences. To overcome this limitation, researchers have explored sparse attention from two core perspectives: position-aware essential computation and content-aware selective computation. The former reduces computational load through predefined sparse patterns, while the latter dynamically determines the scope of computation based on input content. This section reviews the evolution of sparse attention, analyzes the strengths and weaknesses of existing methods, and provides context for our proposed Dynamic Mask Attention.

#### 2.1 Position-Aware Essential Computation

To achieve hardware efficiency, early sparse attention methods predominantly employed fixed sparse patterns, aiming to simplify computation through structured sparsity.

Static Locality. Sliding Window Attention (Beltagy, Peters, and Cohan 2020) confines computation to a local neighborhood for each token, reducing complexity to 𝑂(𝑛 · 𝑤), where 𝑤 is the window size. While simple and efficient, its fixed local window limits the model’s ability to capture long-range dependencies. This limitation is particularly pronounced in tasks requiring information integration across window boundaries, such as long-form question answering or code analysis.

Low-Rank Approximation. Methods like Multi-Head Latent Attention (A. Liu et al. 2024) approximate the full attention matrix using low-rank decomposition to reduce computational and memory demands. While this approach is better at preserving global information than sliding window attention, it comes at the cost of precision loss due to information compression. Low-rank approximation can obscure fine-grained details crucial for specific tasks and, due to its global nature, cannot dynamically adjust its compression strategy based on context, lacking content adaptability.

Hardware-Aligned Sparsity. Work such as Native Sparse Attention (Yuan et al. 2025) designs regularized sparse patterns for modern accelerators, achieving high computational efficiency through hardware-friendly block-sparse structures. However, the core deficiency of such methods lies in their static nature. Fixed sparse patterns cannot adapt to the dynamic changes in input content, leading to potential misallocation of computational resources to non-critical regions while neglecting genuinely important information.

#### 2.2 Content-Aware Selective Computation

As model capabilities have advanced, research has shifted towards content-aware selective computation, enabling models to learn autonomously where to focus their attention.

KV Eviction. Methods like H2O (Zhenyu Zhang et al. 2023) and SnapKV (Y. Li et al. 2024) save memory and computation by evicting "unimportant" tokens from the KV cache. These approaches typically rely on heuristics such as attention scores or access frequency to decide which tokens to retain. While effective in some scenarios, these heuristics can lead to erroneous eviction decisions, permanently losing critical information, especially in complex reasoning tasks that require long-distance backtracking.

Token Selection. Another class of methods actively selects a small subset of tokens for attention computation through techniques like sampling (Z. Chen et al. 2024), hashing (Desai et al. 2024), or clustering (G. Liu et al. 2024). While theoretically appealing, these methods face two major challenges in practice. First, discrete selection operations (such as sampling and hashing) are often non-differentiable, which impedes end-to-end training and prevents the model from learning optimal sparse patterns. Second, token-granular selection strategies disrupt memory access continuity, rendering them incompatible with modern efficient attention implementations like FlashAttention, leading to low hardware utilization and a decrease in both training and inference speed.

Insummary, existingsparseattentionmechanismsfaceaninherenttrade-offbetweenefficiencyandeffectiveness. Positionbased methods, while efficient, lack flexibility and content awareness. Content-based methods, while more intelligent, are often limited by non-differentiable operations and inefficient hardware implementations. This dilemma highlights the urgent need for a new attention paradigm that can leverage structured sparsity for efficient computation while enabling content-aware selection through a trainable, dynamic mechanism.

-----

Inputs

Copying

Outputs

-----

Inputs

Selection

Outputs

-----

Inputs

Induction

Outputs

- Figure 3: Sparsity in Language Modeling Tasks. The tasks of Copy, Select, and Induce are three essential tasks for language modeling. The Copy task requires maintaining a fixed distance between input elements and output elements, the Select task involves selectively remembering or ignoring certain elements based on the input, and the Induce task requires retrieving answers through associative recall based on context. Where the colored parts represent the tokens that the model needs to remember in the current time step of inference, the black parts represent the output tokens that the model needs to predict based on the input, and the white parts represent irrelevant tokens that can be filtered out.

### 3 Background

#### 3.1 Language Modeling Tasks

Sparsity in Language Modeling. As illustrated in Figure 3, long-context language modeling can be decomposed into three fundamental tasks: Copying (Romero et al. 2021), Selecting (Arjovsky, Shah, and Bengio 2016), and Inducing (Olsson et al. 2022). The Copy task requires preserving fixed-distance relationships between input and output tokens. The Select task involves selectively retaining or discarding information based on its content. The Induce task necessitates retrieving information via associative recall from the context. Each of these tasks is characterized by a distinct sparsity pattern: the Copy task exhibits positional sparsity, attending only to tokens at fixed distances; the Select task demonstrates content sparsity, focusing on tokens with specific content; and the Induce task relies on associative sparsity, where attention is directed only to key-value pairs relevant to the query. These inherent sparsity patterns provide a strong theoretical foundation for designing more efficient attention mechanisms.

#### 3.2 Multi-Head Attention

QKV Projection. In the Transformer architecture (Vaswani et al. 2017), the input is first transformed into 𝑄, 𝐾, 𝑉. For the hidden state of the 𝑡-th token in a sequence of length 𝑛, denoted as ℎ𝑡 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙, the linear projections are performed using weight matrices𝑊𝑄, 𝑊𝐾, and𝑊𝑉 to obtain 𝑞𝑡, 𝑘𝑡, and 𝑣𝑡, respectively, as shown in Equation 1. These projections map the input representation into distinct subspaces for each of the 𝑛ℎ attention heads, allowing each head to focus on different aspects of the input. The weight matrices shape the projections to have a dimension of 𝑑ℎ per head.

𝑞𝑡 = ℎ𝑡𝑊𝑄 𝑤ℎ𝑒𝑟𝑒 ℎ𝑡 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙 𝑊𝑄 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙×𝑛ℎ×𝑑ℎ 𝑞𝑡 ∈ R𝑛ℎ×𝑑ℎ 𝑘𝑡 = ℎ𝑡𝑊𝐾 𝑤ℎ𝑒𝑟𝑒 ℎ𝑡 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙 𝑊𝐾 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙×𝑛ℎ×𝑑ℎ 𝑘𝑡 ∈ R𝑛ℎ×𝑑ℎ 𝑣𝑡 = ℎ𝑡𝑊𝑉 𝑤ℎ𝑒𝑟𝑒 ℎ𝑡 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙 𝑊𝑉 ∈ R𝑑𝑚𝑜𝑑𝑒𝑙×𝑛ℎ×𝑑ℎ 𝑣𝑡 ∈ R𝑛ℎ×𝑑ℎ

(1)

Key-Value Concatenation. During autoregressive generation, the key-value pairs of historical tokens are cached to prevent redundant computations. As shown in Equation 2, the cached key and value matrices from past tokens are concatenated with the key-value representations of the current token to form the complete key matrix 𝐾 and value matrix 𝑉. By maintaining and updating this cache, a complete context window spanning all tokens from position 1 to the current position 𝑡 is constructed, enabling the model to access and utilize the full sequence history.

𝑘 = concat([𝑘1, . . .,𝑘𝑡]) 𝑤ℎ𝑒𝑟𝑒 𝑘 ∈ R𝑛ℎ×𝑛×𝑑ℎ 𝑣 = concat([𝑣1, . . .,𝑣𝑡]) 𝑤ℎ𝑒𝑟𝑒 𝑣 ∈ R𝑛ℎ×𝑛×𝑑ℎ

(2)

[Figure 2]

[Figure 3]

Attention Weights Mask Structure

[Figure 4]

[Figure 5]

[Figure 6]

K

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Causal Mask Self Attention

[Figure 14]

[Figure 15]

[Figure 16]

- Q

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

[Figure 27]

[Figure 28]

Dynamic Mask Attention

[Figure 29]

[Figure 30]

nh

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Semi Separable State Space

[Figure 36]

Weights Structure

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Same Structure in Each Head

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Different Structure in Each Head

[Figure 46]

[Figure 47]

=

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Same Structure in Each Head

[Figure 52]

- Figure 4: Dynamic Mask Attention Structure. It demonstrates the mask structure and weight structure of Dynamic Mask Attention in the multi-head case. Unlike the same and redundant mask and weight structures in Self-Attention and State-Space, the mask structure of DMA is dynamically adjusted through content awareness, where each head’s mask can be different. This allows DMA to achieve different attention weight distributions in each head, enabling the model to maximize the utilization of each subspace in multi-head attention and focus on different tokens in each head.

### 4 Method

As discussed in Section 2, existing attention mechanisms are confronted with numerous challenges when processing long sequences, including high computational complexity, substantial memory requirements, the rigidity of static masks, and the presence of non-differentiable components that impede end-to-end training. To address these issues, we introduce Dynamic Mask Attention, a mechanism that strategically leverages the inherent sparsity patterns of language modeling. As illustrated in Figure 2, DMA is composed of two core components: content-aware dynamic sparse masking and positionaware sparse attention weight computation. The former utilizes value representations to dynamically generate masks that determine which historical tokens each attention head should attend to, while the latter performs efficient sparse attention computations guided by these masks. This dual-component design allows DMA to maintain focus on critical information while adapting to varying contextual dependencies. Furthermore, as shown in Figure 4, DMA generates a unique mask structure for each attention head, enabling the model to capture diverse content patterns across different representational subspaces. A sample PyTorch implementation is provided in Listing 1 for reference.

#### 4.1 Generate Content-Aware Dynamic Mask

The content-aware dynamic mask constitutes the central innovation of DMA. It operates by analyzing the content features embedded within the value representations to determine which historical information is relevant to the current query. For each attention head at the current time step, this mechanism generates a unique dynamic mask that directs the subsequent attention weight computation to focus exclusively on the most critical key positions.

To sample the value vector representations, we introduce a sampling weight matrix Δ ∈ R𝑛ℎ×𝑑ℎ×𝑛ℎ, a per-head gating coefficient 𝐴 ∈ R𝑛ℎ, and a non-negative activation function 𝜏(·). As detailed in Equation (3), the process begins with a tensor contraction, 𝑣 · Δ, which projects each token’s 𝑑ℎ-dimensional value vector into a scalar representation, serving

- as an initial estimate of its importance. Subsequently, the activation function 𝜏(·) ensures these scores are non-negative, preventing signal cancellation. The gating coefficient 𝐴 then scales the importance scores for each head, allowing the model to learn varying degrees of sparsity across heads, where 𝐴 can be made query-dependent, enabling the sampling scores to adapt based on the input. Finally, an exponential function amplifies the differences between scores and maps them to a positive value space, which facilitates the learning of gating effects and yields the final scores, 𝛿 ∈ R𝑛ℎ×𝑛.

𝛿 = exp(𝜏(𝑣 · Δ) × 𝐴) 𝑤ℎ𝑒𝑟𝑒 Δ ∈ R𝑛ℎ×𝑑ℎ×𝑛ℎ 𝐴 ∈ R𝑛ℎ 𝛿 ∈ R𝑛ℎ×𝑛 (3)

Subsequently, as defined in Equation (4), a sparsification function 𝑓 (·) is applied. This function identifies whether each score 𝛿𝑛ℎ,𝑗 ranks within the 𝑡𝑜𝑝𝑤 for its respective head. Scores within the 𝑡𝑜𝑝𝑤 are retained to preserve the gradient flow, while all other scores are set to −∞, effectively nullifying their contribution in the subsequent softmax operation. For causal language modeling, this function can incorporate a causal mask via broadcasting to avoid additional memory overhead. This process yields the final dynamic mask, 𝑚𝑡 ∈ R𝑛ℎ×𝑛.

𝑚𝑡 = 𝑓 (𝛿) 𝑤ℎ𝑒𝑟𝑒 𝑚𝑡 ∈ R𝑛ℎ×𝑛





- 𝑓 ( 𝑛𝑗=1 𝛿1,𝑗)
- 𝑓 ( 𝑛𝑗=1 𝛿2,𝑗)

(4)

𝛿𝑛ℎ,𝑗 if 𝛿𝑛ℎ,𝑗 ∈ 𝑡𝑜𝑝𝑤(𝛿𝑛ℎ) −∞ otherwise

𝑤ℎ𝑒𝑟𝑒 𝑓 (𝛿𝑛ℎ,𝑗) =

=

. 𝑓 ( 𝑛𝑗=1 𝛿𝑛ℎ,𝑗)

 

 

This approach offers three significant advantages. First, by sampling importance scores from value representations, the model can more accurately focus on semantically critical tokens, regardless of their distance. This mitigates the risk of overlooking important long-range dependencies, a common issue with methods relying solely on positional patterns. Second, the combination of the gating coefficient 𝐴 and independent 𝑡𝑜𝑝𝑤 selection allows different attention heads to specialize in distinct functions, such as local, long-range, and global, thereby improving the breadth of representational coverage. Third, the sparse selection mechanism is inherently effective during training, eliminating the need for posthoc pruning and preserving the model’s learned retrieval capabilities. The kernel implementation, illustrated in Figure 2 (left), is designed for efficiency. In an outer loop, the sampling weight Δ and gating coefficient 𝐴 are loaded into highspeed SRAM. In an inner loop, a zero-order hold method iteratively processes blocks of the value matrix𝑉 from SRAM to generate content-aware masks for the key matrix 𝐾. These masks are then causally broadcast to match the length of the query matrix 𝑄 in HBM, avoiding memory usage with quadratic complexity. Finally, the mask blocks are concatenated to form the complete content-aware dynamic mask.

#### 4.2 Compute Position-Aware Sparse Weights

The second core component of DMA is the position-aware sparse weight computation. As outlined in Algorithm 1, this component leverages the previously generated dynamic mask to sparsify the computation of scaled dot-product attention weights, effectively reducing the per-step complexity from 𝑂(𝑛𝑑ℎ) to 𝑂(𝑤𝑑ℎ).

For the query at step 𝑡, 𝑞𝑡 ∈ R𝑛ℎ×𝑑ℎ, and the complete key-value pairs, 𝐾,𝑉 ∈ R𝑛ℎ×𝑛×𝑑ℎ, the entire computation flow is detailed in Equation (5). Initially, for each attention head 𝑛ℎ, the scaled dot-product between the query and keys, 𝑞𝑡𝐾⊤, is computed and then element-wise multiplied by the previously constructed dynamic mask 𝑚𝑡. The scaling factor √𝑑ℎ is crucial here as it prevents the dot products from becoming excessively large, which could push the softmax function into a saturated region with minimal gradients. After applying the mask, the softmax function normalizes the results to produce attention weights 𝑝𝑛ℎ,𝑗. Notably, when a mask value𝑚𝑛ℎ,𝑗 = −∞, the corresponding attention weight 𝑝𝑛ℎ,𝑗 ≈ 0, effectively skipping computations for masked positions and filling them with zeros. This ensures that the model focuses solely on relevant unmasked contexts. The attention weights for each head are then multiplied by the value vectors and summed to produce the final context vector 𝑜𝑡 ∈ R𝑛ℎ×𝑑ℎ, where each row captures different contextual patterns and dependencies. The multi-head mechanism, combined with dynamic masking, allows the model to attend to various patterns in parallel across the sequence. This output integrates information from all attention heads, forming a rich hierarchical context representation that effectively captures dependencies at varying distances within the sequence history. It is important to note that this method can approximate full attention when𝑛ℎ×𝑤 ≤ 𝑛, while maintaining computational efficiency.

𝑞𝑡𝑘⊤ √𝑑ℎ ◦𝑚𝑡)𝑣 𝑤ℎ𝑒𝑟𝑒 𝑝𝑡 ∈ R𝑛ℎ×𝑛 𝑜𝑡 ∈ R𝑛ℎ×𝑑ℎ

𝑜𝑡 = softmax(







𝑛

𝑞𝑛ℎ ·𝑘⊤

- 𝑗=1 𝑝1,𝑗 · 𝑣1,𝑗 𝑛
- 𝑗=1 𝑝2,𝑗 · 𝑣2,𝑗

√ 𝑛ℎ,𝑗 𝑑ℎ

exp(

+𝑚𝑛ℎ,𝑗 )

if 𝑚𝑛ℎ,𝑗 ≠ −∞

𝑞𝑛ℎ ·𝑘⊤

𝑤ℎ𝑒𝑟𝑒 𝑝𝑛ℎ,𝑗 =

=

√ 𝑛ℎ,𝑗′

𝑛 𝑗′=1 exp(

+𝑚𝑛ℎ,𝑗′ )

.

𝑑ℎ

 

 

 

0 if 𝑚𝑛ℎ,𝑗 = −∞

𝑛 𝑗=1 𝑝𝑛ℎ,𝑗 · 𝑣𝑛ℎ,𝑗

(5)

- Algorithm 1 Flash Dynamic Mask Attention Forward Pass Per Head Require: Matrices Q, K, V ∈ R𝑁×𝑑ℎ, M ∈ R𝑁 in HBM. Set block sizes 𝐵.

Initialize O = (0)𝑁×𝑑ℎ ∈ R𝑁×𝑑ℎ, ℓ = (0)𝑁 ∈ R𝑁,𝑚 = (−∞)𝑁 ∈ R𝑁 in HBM. Divide Q into𝑇𝑟 blocks of size 𝐵 × 𝑑ℎ each, and divide K, V into 𝑇𝑐 blocks of size 𝐵 × 𝑑ℎ each. Divide M into𝑇𝑐 blocks of size 𝐵 each. Divide O into𝑇𝑟 blocks of size 𝐵 × 𝑑ℎ each, divide ℓ into 𝑇𝑟 blocks of size 𝐵 each, divide 𝑚 into 𝑇𝑟 blocks of size 𝐵 each. for 1 ≤ 𝑗 ≤ 𝑇𝑐 do

Load M𝑗 from HBM to SRAM. Compute active𝑗 = Judge(M𝑗). if active𝑗 = 0 then

Advance stream pointers for K𝑗, V𝑗, continue. end if Load K𝑗, V𝑗 from HBM to SRAM. for 1 ≤ 𝑖 ≤ 𝑇𝑟 do

Load Q𝑖, O𝑖, ℓ𝑖,𝑚𝑖 from HBM to SRAM. Compute S𝑖𝑗 = Q𝑖K𝑇𝑗 × 𝑑ℎ−0.5 + M𝑗 ∈ R𝐵×𝐵. Compute 𝑚˜𝑖𝑗 = rowmax(S𝑖𝑗) ∈ R𝐵, P˜𝑖𝑗 = exp(S𝑖𝑗 −𝑚˜𝑖𝑗) ∈ R𝐵×𝐵, ℓ˜𝑖𝑗 = rowsum(P˜𝑖𝑗) ∈ R𝐵. Compute 𝑚𝑖new = max(𝑚𝑖,𝑚˜𝑖𝑗) ∈ R𝐵, ℓ𝑖new = 𝑒𝑚𝑖−𝑚𝑖newℓ𝑖 + 𝑒𝑚˜𝑖𝑗−𝑚𝑖newℓ˜𝑖𝑗 ∈ R𝐵. Write O𝑖 ← diag(ℓ𝑖new)−1(diag(ℓ𝑖)𝑒𝑚𝑖−𝑚𝑖newO𝑖 + 𝑒𝑚˜𝑖𝑗−𝑚𝑖newP˜𝑖𝑗V𝑗) to HBM. Write ℓ𝑖 ← ℓ𝑖new, 𝑚𝑖 ← 𝑚𝑖new to HBM.

end for end for Return O, ℓ,𝑚.

This method offers three key advantages. First, the mask prunes the set of candidate tokens before the matrix multiplication and softmax operations. This avoids the inefficiency of pseudo-sparsity, where computations are performed for all tokens only to be zeroed out afterward. Second, unlike methods that perform key-value selection by discarding tokens, our approach preserves the full sequence. This ensures that the complete global context remains available for all attention heads to access as needed. Third, the kernel implementation can perform block-level skipping by loading a block of the mask to check if all positions within it are masked. If so, the entire block is skipped, avoiding unnecessary memory loads and matrix multiplication operations. The kernel implementation, depicted in Figure 2 (right), is optimized for this process. In an outer loop, blocks of the 𝐾 and 𝑉 matrices are loaded into SRAM. In an inner loop, blocks of the 𝑄 matrix are loaded, and if the corresponding 𝐾 block is not entirely masked, the attention weights are computed and the output is written back to HBM. If a position in the 𝐾 block is masked, its attention weight is set to zero, and the computation for that position is skipped, resulting in position-aware sparse attention weights.

#### 4.3 Fully Gradient Flow

Finally, the entire backward process is outlined in Algorithm 2. We ensure that the introduced dynamic mask and sparse weights do not block gradients, and the gradients of the retained attention paths are strictly consistent with those of full attention. They can flow completely to all inputs and parameters without gradient discontinuity issues caused by discrete operations, supporting end-to-end training while minimizing redundant costs.

For clarity, our derivation considers a single attention head ℎ at a single time step 𝑡. Let Iℎ ⊂ {1, . . .,𝑛} be the set of 𝑤 indices selected for this head. For unselected indices 𝑗 ∉ Iℎ, the mask value is treated as𝑚ℎ,𝑗 = −∞. The key intermediate variables in the forward pass are defined in Equation (6).

√𝑑ℎ +𝑚ℎ,𝑗 𝑝ℎ,𝑗 =   

exp(𝑠ℎ,𝑗) 𝑗′∈Iℎ exp(𝑠ℎ,𝑗′)

𝑜ℎ = ∑︁ 𝑗∈Iℎ

𝑗 ∈ Iℎ 0 𝑗 ∉ Iℎ

𝑞ℎ · 𝑘ℎ,𝑗

𝑝ℎ,𝑗𝑣ℎ,𝑗 (6)

𝑠ℎ,𝑗 =

In the backward pass, let the upstream gradient of the loss function 𝐿 with respect to the head’s output 𝑜ℎ be 𝑑𝑜ℎ = 𝜕𝑜𝜕𝐿

∈

ℎ

- R𝑑ℎ. As shown in Equation (7), the gradient for 𝑣 is computed by distributing𝑑𝑜ℎ to the selected vectors 𝑣ℎ,𝑗 in proportion to their attention weights 𝑝ℎ,𝑗, while the gradients for unselected positions are zero.

𝜕𝐿 𝜕𝑣ℎ,𝑗

=

𝑝ℎ,𝑗 𝑑𝑜ℎ 𝑗 ∈ Iℎ 0 𝑗 ∉ Iℎ

(7)

- Algorithm 2 Flash Dynamic Mask Attention Backward Pass Per Head

Require: Matrices Q, K, V, O, dO ∈ R𝑁×𝑑ℎ, M, dM ∈ R𝑁 in HBM, vectors ℓ,𝑚 ∈ R𝑁 in HBM. Set block sizes 𝐵. Divide Q into𝑇𝑟 blocks of size 𝐵 × 𝑑ℎ each, and divide K, V in to 𝑇𝑐 blocks of size 𝐵 × 𝑑ℎ each. Divide M into𝑇𝑐 blocks of size 𝐵 each. Divide O, dO into 𝑇𝑟 blocks of size 𝐵 × 𝑑ℎ each, divide ℓ into 𝑇𝑟 blocks of size 𝐵 each, and divide 𝑚 into 𝑇𝑟 blocks of size 𝐵 each. Initialize dQ = (0)𝑁×𝑑ℎ in HBM and divide it into 𝑇𝑟 blocks of size 𝐵 × 𝑑ℎ each. Initialize dK = (0)𝑁×𝑑ℎ, dV = (0)𝑁×𝑑ℎ in HBM and divide dK, dV in to 𝑇𝑐 blocks of size 𝐵 × 𝑑ℎ each. Initialize dM = (0)𝑁 in HBM and divide it into 𝑇𝑐 blocks of size 𝐵 each. for 1 ≤ 𝑗 ≤ 𝑇𝑐 do

Load M𝑗 from HBM to SRAM. Compute active𝑗 = Judge(M𝑗). if active𝑗 = 0 then

Advance stream pointers for K𝑗, V𝑗, continue. end if Load K𝑗, V𝑗 from HBM to SRAM. Initialize dK˜ 𝑗 = (0)𝐵×𝑑ℎ, dV˜ 𝑗 = (0)𝐵×𝑑ℎ, dM˜ 𝑗 = (0)𝐵 on SRAM. for 1 ≤ 𝑖 ≤ 𝑇𝑟 do

Load Q𝑖, O𝑖, dO𝑖, dQ𝑖, ℓ𝑖,𝑚𝑖 from HBM to SRAM. Compute S𝑖𝑗 = Q𝑖K𝑇𝑗 × 𝑑ℎ−0.5 + M𝑗 ∈ R𝐵×𝐵. Compute P𝑖𝑗 = diag(ℓ𝑖)−1 exp(S𝑖𝑗 −𝑚𝑖) ∈ R𝐵×𝐵. Compute dV˜ 𝑗 ← dV˜ 𝑗 + P𝑖𝑗⊤dO𝑖 ∈ R𝐵×𝑑ℎ. Compute dP𝑖𝑗 = dO𝑖V⊤𝑗 ∈ R𝐵×𝐵. Compute 𝐷𝑖 = rowsum(dO𝑖 ◦ O𝑖) ∈ R𝐵. Compute dS𝑖𝑗 = P𝑖𝑗 ◦ (dP𝑖𝑗 − 𝐷𝑖) ∈ R𝐵×𝐵. Compute dM˜ 𝑗 ← dM˜ 𝑗 + rowsum(dS𝑖𝑗) ∈ R𝐵. Write dQ𝑖 ← dQ𝑖 + dS𝑖𝑗K𝑗 × 𝑑ℎ−0.5 ∈ R𝐵×𝑑ℎ to HBM. Compute dK˜ 𝑗 ← dK˜ 𝑗 + dS𝑖𝑗⊤Q𝑖 × 𝑑ℎ−0.5 ∈ R𝐵×𝑑ℎ.

end for Write dK𝑗 ← dK˜ 𝑗, dV𝑗 ← dV˜ 𝑗, dM𝑗 ← dM𝑗 + dM˜ 𝑗 to HBM.

end for Return dQ, dK, dV, dM.

Next, we compute the gradient for the scores 𝑠ℎ,𝑗. The gradient of the attention weights 𝑝ℎ,𝑗 with respect to their inputs is 𝑑𝑝ℎ,𝑗 = 𝑣ℎ,𝑗 ·𝑑𝑜ℎ. Using the standard softmax Jacobian, we can derive the gradient for 𝑠ℎ,𝑗, denoted as 𝑑𝑠ℎ,𝑗, as shown in Equation (8). For masked positions where 𝑝ℎ,𝑗 = 0, the gradient 𝑑𝑠ℎ,𝑗 is naturally zero.

𝑑𝑠ℎ,𝑗 = 𝑝ℎ,𝑗 (𝑑𝑝ℎ,𝑗 − ∑︁

𝑝ℎ,𝑗′ × 𝑑𝑝ℎ,𝑗′) (8)

𝑗′∈Iℎ

Because 𝑠ℎ,𝑗 is an additive combination of 𝑞ℎ ·𝑘ℎ,𝑗 and 𝑚ℎ,𝑗, the gradient is distributed directly. The gradient for the mask 𝑚ℎ,𝑗 is simply 𝑑𝑠ℎ,𝑗, as shown in Equation (9). This ensures that gradients can flow directly to Δ and 𝐴.

𝜕𝐿 𝜕𝑚ℎ,𝑗

= 𝑑𝑠ℎ,𝑗 (9)

Finally, asshown inEquation(10), the gradientsfor𝑞ℎ and𝑘ℎ,𝑗 are obtained bybackpropagating𝑑𝑠ℎ,𝑗 throughthe computation path. Crucially, the gradient calculations only involve the selected index set Iℎ, thereby reducing computation.

= ∑︁

𝑘ℎ,𝑗 √𝑑ℎ

𝑞ℎ √𝑑ℎ

𝜕𝐿 𝜕𝑘ℎ,𝑗

𝜕𝐿 𝜕𝑞ℎ

(10)

= 𝑑𝑠ℎ,𝑗

𝑑𝑠ℎ,𝑗

,

𝑗∈Iℎ

Our approach has several significant advantages. First, for the selected positions, the gradients are identical to those of full attention, and DMA only prunes the operator chain for positions whose contributions can be ignored, ensuring expressiveness. Then, only second-order correlation information is propagated to Iℎ, improving bandwidth utilization. The gating parameter 𝐴 and weight Δ directly receive attention weights as gradients, quickly shaping head specialization. Finally, the equivalence relation 𝑑𝑀 = 𝑑𝑆 allows the kernel to only recompute the local 𝑆 without storing additional intermediate mask gradient tensors.

Scaling Laws on The SmolLMCorpus

|MHA SWA MLA NSA DMA<br><br>|
|---|

- 2 × 101
- 3 × 101

Perplexity

101

5 × 100

- 2 × 100
- 3 × 100

1019 1020 1021

FLOPs

- Figure 5: Scaling Laws. The perplexity performance of different self-attention variants on SmolLMCorpus at different parameter scales. For suboptimal variants like SWA and MLA, we omit them for clarity. Compared to other variants, our Dynamic Mask Attention has a Pareto advantage in performance.

### 5 Experiments

We will validate the efficiency and effectiveness of Dynamic Mask Attention, as detailed in Section 4, in handling long contexts through its content-aware dynamic sparse mask and position-aware dynamic sparse weight computation.

#### 5.1 Experimental Settings

Baselines. To thoroughly evaluate DMA, we benchmarked it against representative baselines surveyed in Section 2. First, we compared DMA with various mainstream attention variants in terms of pre-training perplexity at different model scales, further validating DMA’s advantage in long-sequence information retrieval through the challenging multiquery associative recall task, and tested the efficiency of our kernel compared to other optimized implementations in

- Section 5.2. Second, we compared DMA, NSA, and MHA on a 1.7B parameter Transformer model pre-trained and supervisedly fine-tuned on 40B tokens, evaluating them on downstream benchmark tasks and needle-in-a-haystack tests in
- Section 5.3.

Training Settings. All experiments were conducted using the open-source PyTorch images (NVIDIA 2022) and the Transformers framework (Wolf et al. 2020). For model configuration, we consistently employed the NeoX tokenizer (Black et al. 2022), the AdamW optimizer (Loshchilov and Hutter 2017), and the WSD learning rate scheduler (Hägele et al. 2024), while strictly adhering to the Optimal Hyperparameter Scaling Law (H. Li et al. 2025) and the Chinchilla (Hoffmann et al. 2022) standard protocol throughout our training on the SmolLMCorpus (Ben Allal et al. 2024) dataset. For evaluation frameworks, we utilized the LM evaluation harness (L. Gao et al. 2021) from EleutherAI for perplexity tasks, and the lighteval (Fourrier et al. 2023) from HuggingFace for downstream tasks.

#### 5.2 Variants Comparison

Scaling Perplexity. First, we present the comparison of the perplexity performance of different self-attention variants

- at various parameter scales in Figure 5. This experiment includes the baseline 1, sliding window attention 2 driven by static mask structures, multi-head latent attention 3 driven by low-rank decomposition approximations, native sparse attention 4 driven by hardware content adaptation, and our proposed Dynamic Mask Attention. These experiments were conducted on the SmolLMCorpus dataset, with model sizes ranging from 80M to 1.7B parameters, and the experimental configurations are detailed in Table 4. Our experimental results validate that Dynamic Mask Attention maintains the best performance across various scales. We speculate that this advantage primarily stems from DMA’s ability to adaptively focus on key information in the input sequence, effectively avoiding the lost in middle (N. F. Liu et al. 2024) problem.

MHA SWA MLA NSA DMA

Sequence Length: 1024

Sequence Length: 2048

Sequence Length: 4096

Sequence Length: 8192

1.00

0.75

Accuracy

0.50

0.25

0.00

32 64 128 256 Model Dimension

32 64 128 256 Model Dimension

32 64 128 256 Model Dimension

32 64 128 256 Model Dimension

- Figure 6: Multi-Query Associative Recall. This is a more challenging version of the original multi-query associative recall task (Arora et al. 2024), which includes longer sequence lengths and smaller model dimensions. Dynamic Mask Attention maintains good performance in most cases.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

256 5121,0242,0484,0968,19216,38432,768

- 100
- 101
- 102

Forward

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

2565121,0242,0484,0968,19216,38432,76865,536131,072262,144524,288

- 100
- 101
- 102

Decode

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

256 5121,0242,0484,0968,19216,38432,768

- 100
- 101
- 102

Backward

Sequence length

Runtime(ms)

MHA SWA MLA NSA DMA

- Figure 7: Kernal Performance. A performance comparison of efficient implementation kernels of different attention variants on an A100 GPU. Our DMA achieves significant acceleration in forward propagation, decoding, and backward propagation, maintaining the same efficiency level as SWA.

Associative Recall. To further validate the ability of different attention variants in long-sequence information retrieval, we designed a more challenging variant of the multi-query associative recall task (Arora et al. 2024), which includes longer sequence lengths and smaller model dimensions. This task assesses the ability of language models to retrieve information within their context. Specifically, it provides key-value pairs to the autoregressive model, prompting the model to generate the correct value when displaying previously seen keys. To increase the difficulty of the task, we used 512 key-value pairs in the experiment. We employ sliding window attention, native sparse attention, and dynamic mask attention, all with a window size of 512. This approach replaces non-query/key/value parts with random tokens, forcing the model to locate relevant information precisely rather than relying on contextual clues. The experimental dataset comprises 250,000 training samples and 1,000 test samples, with all models trained for 100 epochs to ensure sufficient convergence. As shown in Figure 6, Dynamic Mask Attention performs excellently across various sequence lengths, indicating its ability to intelligently identify and focus on tokens relevant to the current state while ignoring irrelevant tokens.

Kernel Acceleration. To analyze the performance of DMA within modern efficient operator frameworks, we benchmarked the forward, decoding, and backward performance of MHA, SWA, MLA, NSA, and DMA on an A100-SXM4-80GB GPU. The results represent the average of 1,000 runs after three warm-up iterations; specific configurations and implementations can be found in Table 5. As shown in Figure 7, for the Forward pass, compared to MHA, DMA achieves speedups of approximately 26.1×, 10.2×, and 21.5× at token lengths of 8192, 16384, and 32768, respectively. For the Decode phase, speedups against MHA at key lengths of 65536, 131072, 262144, and 524288 are approximately 49.6×, 92.7×, and 171.1×, respectively. For the Backward pass, speedups against MHA at lengths of 8192, 16384, and 32768 are approximately 2.5×,

- 4.4×, and 7.9×, respectively. DMA avoids large-scale redundant score and softmax backpropagation overhead by explicitly skipping masked blocks, thus providing strong acceleration capabilities across multiple critical stages.

Multi Head Attention

Native Sparse Attention

Dynamic Mask Attention

1.0

0% 10% 20% 30% 40% 50% 60% 70% 80% 90%

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

[Figure 53]

0.8

DepthPercent

0.6

Score

0.4

0.2

100%

0.0

1K 2K 3K 4K 6K 8K 10K 12K 14K 16K

1K 2K 3K 4K 6K 8K 10K 12K 14K 16K

1K 2K 3K 4K 6K 8K 10K 12K 14K 16K

Token Limit

- Figure 8: Needle in a Haystack. Comparison of needle-in-a-haystack performance between MHA, NSA, and DMA in an apples-to-apples setting. The white dotted line indicates the sequence length of the model.

- Table 2: Downstream Task Evaluations for Base Model. The best results for each size are in bold, and the second-best results are unlined. DMA outperforms MHA and NSA, as well as other advanced inference sparse methods, on most tasks.

Model LAMBADA LAMBADA MMLU TriviaQA ARC PIQA HellaSwag OBQA WinoGrande Avg ppl ↓ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑

Zero-Shot

MHA 15.22 44.3 35.4 9.4 53.4 72.9 56.1 37.0 57.3 45.7 H2O 15.38 44.2 34.8 7.4 53.3 72.8 55.6 36.6 56.9 45.2

InfLLM 15.23 44.2 35.1 8.0 53.1 72.4 55.8 36.6 56.8 45.2 Quest 15.43 43.9 35.1 7.6 53.1 72.6 56.1 36.8 57.2 45.3 DAM 15.89 44.5 34.6 8.9 52.1 72.3 56.2 36.3 56.0 45.1

Exact-Top 15.23 44.4 35.3 9.2 53.3 72.8 56.0 36.8 57.0 45.6 NSA 14.91 45.2 33.8 8.7 53.1 72.8 56.7 36.3 57.8 45.5 DMA (ours) 14.42 45.9 37.0 9.1 55.6 73.4 56.4 36.5 58.4 46.5

Five-Shot

MHA 19.40 40.4 36.8 13.2 56.8 73.2 56.8 38.0 58.6 46.7 H2O 19.14 38.9 35.7 10.2 56.6 73.2 56.4 37.8 58.1 45.8

InfLLM 19.13 41.3 35.9 11.7 56.7 73.3 56.1 38.0 57.7 46.3 Quest 19.22 40.9 36.1 10.9 56.2 73.2 55.8 37.9 58.2 46.1 DAM 19.47 41.2 35.2 13.3 55.1 71.0 54.4 38.0 57.2 45.6

Exact-Top 18.22 39.7 36.4 13.1 56.3 73.4 56.5 38.2 58.5 46.5 NSA 21.37 39.6 34.6 12.5 56.1 76.0 58.9 39.2 58.3 46.9 DMA (ours) 17.88 40.9 38.2 12.6 56.4 76.6 58.7 39.6 60.4 47.9

#### 5.3 Performance Comparison

Downstream Benchmark Evaluations for Base Model. We used the Qwen3 1.7B (Team 2025) model structure as a baseline, making only modifications to the self-attention part for comparison. We first pre-trained the model on a high-quality dataset covering four domains: Web, TextBook, Code, and Math, with a total of 32 billion tokens and a sequence length of 2,048, thereby providing the model with basic language skills and general knowledge. Subsequently, we carefully selected 8B tokens packaged into sequences of length 8K. We conducted a second phase of pre-training by adjusting the RoPE base frequency from 10K to 100K (Xiong et al. 2023), ensuring that the model could effectively handle longer inputs. Ultimately, we obtained three models: MHA, NSA, and DMA, and evaluated them on the following tasks: LLAMBADA (Paperno et al. 2016), MMLU (Hendrycks et al. 2021), TriviaQA (Joshi et al. 2017), ARC (P. Clark et al. 2018), PIQA (Bisk et al. 2020), HellaSwag (Zellers et al. 2019), OBQA (Mihaylov et al. 2018), Winogrande (Sakaguchi et al. 2021), and the English tasks of LongBench (Bai et al. 2023). We also compared several advanced inference sparse methods, including H2O (Zhenyu Zhang et al. 2023), infLLM (C. Xiao et al. 2024), Quest (Jiaming Tang et al. 2024), DAM (Hanzhi Zhang et al. 2025), and Exact-Top, which first computes full attention scores using MHA and then performs sparsification based on that. The results are shown in Table 2. In both zero-shot and five-shot settings, DMA outperforms the baseline on most tasks, achieving excellent overall performance.

- Table 3: Downstream Evaluations for Finetuned Model. The best results are in bold, and the second-best results are underlined. For models supervisedly fine-tuned at a 16K sequence length, DMA outperforms other methods in most tasks.

Method MMLU BBH GSM8K MATH MBPP ARC PIQA LongBench RULER Avg acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑

MHA 46.4 37.7 46.3 11.7 40.0 59.8 76.2 30.2 60.6 39.0 H2O 42.4 34.5 44.8 10.5 38.5 57.6 74.1 28.5 47.8 35.3

InfLLM 44.2 36.0 45.5 11.0 39.2 58.9 75.6 29.1 55.3 37.2 Quest 44.0 34.8 45.0 10.8 38.8 56.2 73.7 28.9 50.7 36.1 DAM 45.0 36.5 46.0 11.5 37.2 56.9 72.4 26.4 49.2 36.0

Exact-Top 45.1 37.2 45.1 11.5 38.4 57.3 75.4 28.3 44.8 35.7 NSA 43.8 38.4 46.2 11.6 40.5 59.1 75.7 30.2 59.6 38.6 DMA (ours) 46.2 38.2 46.8 11.6 40.6 59.6 76.6 30.7 60.5 39.2

Downstream Benchmark Evaluations for Finetuned Model. We further fine-tuned all models at a 16K sequence length with an adjusted RoPE base frequency to 400K, to further enhance the models’ long-context generalization capabilities. Ultimately, we obtained three fine-tuned models: MHA, NSA, and DMA, and evaluated them on the following tasks: MMLU (Hendrycks et al. 2021), BBH (Suzgun et al. 2023), GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2020), MBPP (Austin et al. 2021), LongBench (Bai et al. 2023), and RULER (Hsieh et al. 2024). We used the same advanced inference sparse methods, which also compute full attention scores using MHA and then perform sparsification based on that. The results are shown in Table 3. DMA achieved the best average score, leading in GSM8K, MBPP, and LongBench, while remaining highly competitive in MMLU and RULER. NSA ranked first in BBH, with DMA closely following, while the full-attention MHA performed best in RULER but lagged behind DMA in average score. These results indicate that DMA’s content-aware sparse mask effectively transfers even under longer context fine-tuning.

Extrapolated Content Retrieval. We further conducted an apples-to-apples comparison between MHA, NSA, and DMA using the needle-in-a-haystack task (Kamradt 2023) to evaluate the models’ ability to retrieve information accurately from long texts. In this synthetic retrieval task, a random and information-rich sentence is inserted into a lengthy document, and the model needs to retrieve the needle from the haystack to answer the question. As shown in Figure 8, as the context length increases, the advantage of DMA over NSA and MHA gradually expands. Notably, when the context length exceeds the pre-training sequence length, all three models exhibit a performance decline; however, the decrease in DMA’s performance is significantly smaller than that of NSA and MHA, demonstrating stronger extrapolation capabilities and more effective retrieval of information in unseen length ranges. We speculate that trainable sparse attention inherently possesses stronger sequence length extrapolation. This experimental result has dual significance: on one hand, it validates DMA’s intrinsic advantages in handling ultra-long documents, especially in practical application scenarios that require precise localization and extraction of key information; on the other hand, it reveals the structural advantages of DMA’s content-aware dynamic mask mechanism in maintaining long-distance dependency modeling capabilities, even when the sequence length exceeds the pre-training range, thus maintaining relatively stable performance. This extrapolation capability is of great value for practical applications that require processing long documents.

Our comprehensive experimental results demonstrate the exceptional performance of Dynamic Mask Attention across various tasks and model scales. In scaling perplexity experiments, DMA consistently outperformed other attention variants across different parameter scales from 80M to 1.7B; in the multi-query associative recall task, DMA exhibited superior information retrieval capabilities and efficiency; in kernel implementations, DMA showed extremely high speedup ratios in various long-sequence application scenarios; in downstream benchmark evaluations, DMA models outperformed the original MHA and its various sparse variants on most tasks; in the needle-in-a-haystack task, DMA demonstrated significantly stronger length extrapolation capabilities. These results collectively validate the effectiveness of DMA as a sparse attention solution that simultaneously enhances computational efficiency and model performance.

Pocess

Pocess

Pocess

Pocess

Full Attention Dynamic Mask Attention

Causal Head Pattern Local Context Head Pattern Range Dependency Head Pattern Global Context Head Pattern

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Process Process Process Process

- Figure 9: Weights Heatmaps of DMA. The heatmaps show the attention weights of each head in the Dynamic Mask Attention mechanism, indicating which tokens each head focuses on. Full heatmaps can be found in Appendix C.

0 100 200 300 400 500

1.0

0.5

0.0

0.5

1.0

1.5

2.0

Tokenvalue

Adaptive filtering keeps informative tokens while skipping noise

Noisy input signal

Underlying structure

Static sampling

Adaptive selection

0 100 200 300 400 500

0.00

0.25

0.50

0.75

1.00

1.25

1.50

Score/Mask

Dynamic mask behaves like a content-aware filter

Content saliency

Selection threshold

Adaptive mask

Uniform sampling

- Figure 10: Adaptive Filtering of DMA. Top: the noisy token signal retains its underlying low-frequency structure while the adaptive mask focuses on informative tokens compared with uniform sampling. Bottom: the learned content-aware scores and the resulting mask illustrate how DMA allocates compute to relevant regions while skipping redundant context.

### 6 Analysis

In this section, we analyze Dynamic Mask Attention, highlighting its distinct advantages in handling long-range dependencies and providing dynamic context awareness.

#### 6.1 Head Specialization

As shown in Figure 9 11, our analysis of the attention patterns learned by the model reveals how DMA creates contentaware sparse structures that adapt to different contextual needs. Unlike the uniform patterns of traditional attention mechanisms, each DMA attention head develops a unique sparse pattern: some heads focus on the most recent tokens to capture local context, while others attend to specific distant positions for long-range dependencies, and additional heads maintain broader context awareness for global understanding. This diversity allows the model to capture various types of dependencies simultaneously while maintaining computational efficiency, making efficient use of each subspace.

14

Local Context Heads. For example, Head 0, Head 1, Head 8, Head 10, and Head 11 tend to focus on the most recent tokens, forming a local band attention pattern. These heads are primarily responsible for capturing syntactic structures, phrase-level semantics, and local dependencies, which are particularly important for tasks requiring precise local context handling.

Range Dependency Heads. For example, Head 2, Head 3, Head 4, Head 5, and Head 14 demonstrate the ability to attend to specific distant tokens. These heads are specialized in capturing long-range semantic associations, such as resolving coreference issues or tracking complex storylines in lengthy documents. They can skip over large amounts of intermediate information and directly connect distant but semantically related parts, which is crucial for deep reasoning and contextual understanding.

Global Context Heads. For example, Head 6, Head 7, Head 9, Head 12, Head 13, and Head 15 exhibit a sparser but broader attention distribution, sampling key information from the entire sequence to form an overall perception of the global context. These heads function similarly to summarizers, responsible for integrating information from different parts to create a coherent global representation. This capability is crucial for tasks that require a comprehensive understanding of the entire input to make accurate predictions.

Dynamic Adaptability. The most significant advantage of DMA lies in its dynamism. These attention patterns are not static; they are dynamically generated based on the input content. This means the model can adjust its attention strategy in real-time, activating the most appropriate combination of heads when processing different tasks or text types. For example, when processing code, it might rely more on long-range dependency heads to track variable definitions and usages, whereas in a conversation, it might focus more on local context heads to understand the current exchange. This content-aware adaptability is the core advantage of DMA over static sparse attention methods.

This naturally occurring specialization is a direct result of the content-aware mask mechanism, enabling the model to effectively handle various types of dependencies while maintaining computational efficiency, achieving effective integration of multi-scale information. This hierarchical integration mechanism can effectively handle multi-level semantic structures in complex texts. It is worth noting that head specialization may also occur in traditional MHA, but the specialization patterns in DMA are more pronounced and functionally clearer, which may be a key reason for its superior performance across various tasks.

#### 6.2 Adaptive Filtering

From the perspective of modern signal processing, Trainable Dynamic Mask Sparse Attention essentially performs dynamic downsampling of the input sequence through learnable adaptive filters, i.e., masks, retaining only key information components. This allows for efficient extraction of low-frequency dependencies in long-distance signals, such as text, while suppressing noise redundancy. The core logic is to treat long texts as noisy low-frequency signals, where the mask acts as an adaptive filter, and the dynamically selected retained tokens are equivalent to intelligent downsampling based on signal relevance.

Learnable Content-aware Filters. Unlike traditional sparse methods that rely on fixed patterns, DMA’s mask is dynamically generated based on the input content, making it a content-aware adaptive filter. This filter learns to identify and amplify key components in the signal, i.e., important tokens, while attenuating or completely filtering out noise, i.e., irrelevant tokens. This mechanism ensures that computational resources are precisely allocated to the most critical information for the current task, effectively avoiding information loss due to excessively long contexts in needle-in-a-haystack problems.

Multi-scale Signal Decomposition. The different attention heads in DMA learn various sparse patterns, which can be viewed as a set of parallel adaptive filters, each responsible for capturing different scales or types of signal features. This multi-scale decomposition allows the model to build a comprehensive and hierarchical understanding of the input signal with extremely high efficiency.

Recasting sparse attention as an adaptive filtering problem, DMA offers a new perspective for understanding and optimizing long text processing. It achieves intelligent filtering of information through content awareness and multi-scale decomposition, ensuring that the model learns the optimal sparse strategies.

### 7 Discussion

In this section, we discuss the core deficiencies of existing sparse attention methods, analyze how Dynamic Mask Attention addresses these issues, and explore its limitations and future development directions.

- 7.1 Limitations of Existing Approaches Existing sparse attention methods exhibit three critical deficiencies that limit their practical effectiveness:

Post-hoc Sparsification Degradation. The performance degradation caused by post-hoc sparsification stems from the fundamental mismatch between existing methods and the optimization trajectory of pretrained models. As demonstrated by Chen et al. (Z. Chen et al. 2024), retaining only the top 20% of attention weights covers only 70% of the total attention scores. This forced sparsification strategy compels models to deviate from the optimal parameter configurations learned on large-scale corpora. More critically, this approach causes irreversible damage to key structural components in pretrained models, such as retrieval heads and copy heads, as these specialized attention heads are misidentified as "unimportant" and pruned during inference. This structural destruction directly leads to significant performance degradation in tasks that require precise information retrieval and copying.

Training-Inference Efficiency Gap. Most sparse attention methods optimize only for inference, neglecting trainingphase computational demands. This creates bottlenecks across LLM development: pretraining on long documents, longcontext fine-tuning, and reinforcement learning. Without effective training-time sparsity support, these crucial phases remain constrained by𝑂(𝑛2) computational complexity, limiting development of more capable long-context models.

Non-differentiable Components and Inefficient Backpropagation. Non-differentiable components and inefficient backpropagation problems reveal the technical shortcomings of existing methods in terms of trainability. The discrete operations in methods like ClusterKV (G. Liu et al. 2024) and MagicPIG (Z. Chen et al. 2024) introduce discontinuities in computational graphs, which block gradient flow and hinder the learning of optimal sparse patterns. Even trainable methods like HashAttention (Desai et al. 2024) suffer from memory access inefficiencies due to token-granular selection, which is incompatible with the contiguous memory access and block-wise computation requirements of efficient attention techniques, such as FlashAttention. Consequently, these implementations are forced to revert to naive implementations with low hardware utilization, significantly degrading training efficiency.

- 7.2 How Dynamic Mask Attention Addresses Core Issues

DynamicMaskAttentionsystematicallyaddressestheaforementionedfundamentalissuesthroughthreecoreinnovations, achieving unified, efficient, and sparse computation for both training and inference phases.

Native Trainable Sparsity. Native trainable sparsity is DMA’s key innovation for addressing post-hoc sparsification issues. Unlike traditional methods, DMA embeds sparsity into the model architecture from the ground up, ensuring that sparse attention patterns are fully aligned with the model’s optimization trajectory. Specifically, DMA retains complete, uncompressed KV caches 𝑘 = concat([𝑘1, . . .,𝑘𝑡]) and 𝑣 = concat([𝑣1, . . .,𝑣𝑡]), ensuring the original fidelity of historical information and precise recall capabilities, avoiding information bottlenecks that may arise from fixed-state compression in State Space Models. This comprehensive information retention mechanism enables DMA to precisely access any token in the historical context at any moment, without losing critical information due to lossy compression methods like Mamba. More importantly, DMA’s sparsification occurs during the attention weight computation phase, rather than in post-training processing, ensuring that models do not deviate from pre-trained parameter configurations during sparsification, thereby protecting key structural components, such as retrieval heads and copy heads, from damage.

Unified Training-Inference Architecture. The unified training-inference architecture eliminates the fundamental gap in training-inference efficiency that exists in existing methods. DMA’s dynamic weight computation 𝛿 = exp(𝜏(𝑣Δ) × 𝐴) uses identical sparsification strategies during both training and inference phases. This consistency ensures that models can learn optimal sparse patterns during training and seamlessly apply these patterns during inference. This unified architecture particularly benefits three critical stages of modern LLM development: the pretraining stage can efficiently process long document sequences; the long-context fine-tuning stage can adapt to specific task requirements; the reinforcement learning stage can effectively update attention weights through policy gradients. DMA reduces computational complexity from 𝑂(𝑛2) to 𝑂(𝑛 · 𝑤), enabling the training of larger-scale long-context models.

Fully Differentiable Design. The fully differentiable design ensures that DMA maintains gradient flow continuity throughout the entire computation process. The computation of dynamic mask weights 𝛿 is based entirely on differentiable operations, including linear transformations of value representations, non-negative activation functions 𝜏(·), and exponential functions, thereby avoiding gradient interruptions caused by discrete operations such as k-means clustering and SimHash. Although the mask generation process involves a top-k operation, it is not the core learning objective of DMA but merely a tool for sparse selection; thus, we only use this discrete operation in the forward pass. Moreover, the attention weight computation part is designed such that the gradients for masked positions should naturally be zero, so skipping computation and setting gradients to zero is the correct behavior. This design enables the model to learn optimal attention patterns that are sparse in an end-to-end manner, dynamically adjusting which historical positions are most critical for current reasoning, thereby achieving truly content-aware, selective computation. Additionally, each head in a multi-head attention mechanism can independently generate different sparse patterns, thereby the representational capabilities of the multi-head architecture by focusing on different information segments in distinct subspaces.

#### 7.3 Limitations and Future Works

Despite Dynamic Mask Attention’s significant progress in addressing the core issues of existing methods, several limitations remain that warrant further exploration and improvement in future work.

Adaptive Window Size Selection. Adaptive window size selection is the primary challenge facing DMA. While the current fixed window size design provides predictable computational complexity, it may not optimally adapt to the dynamic demands of different tasks and contexts. For instance, code generation tasks may require larger windows to capture longrange structural dependencies, while simple question-answering tasks may only need smaller windows. Future research directions include developing adaptive window size selection mechanisms based on task complexity, sequence length, and content features, potentially through reinforcement learning or meta-learning approaches to dynamically optimize window parameters. Alternatively, designing hierarchical multi-scale attention structures can be considered to capture dependencies across different ranges simultaneously.

Position Encoding Enhancement. Our needle-in-a-haystack experiments revealed an intriguing phenomenon: trainable sparse attention mechanisms, such as DMA, exhibit stronger length extrapolation capabilities compared to dense attention when context lengths exceed the pretraining bounds. This finding suggests that the fundamental bottleneck for extrapolation may lie in the position encoding method rather than the attention mechanism itself. Current RoPE-based position encodings struggle with out-of-distribution sequence lengths, but DMA’s dynamic sampling architecture offers a potential alternative pathway for encoding positional information. Specifically, the zero-order hold sampling values that are added as attention biases can be explored to explicitly incorporate positional information into these sampling values, potentially replacing or complementing RoPE to create a more extrapolation-friendly encoding scheme. Such an approach might leverage the inherent advantages of sparse attention’s selective computation to create position representations that scale more naturally to unseen lengths. This direction could help address one of the most persistent challenges in longcontext modeling: maintaining consistent positional understanding across arbitrary sequence lengths without requiring length-specific fine-tuning.

Multi-Modal Extension. Multi-modal extension represents an essential direction for DMA development. The current DMA design is primarily optimized for text sequences; however, modern AI systems increasingly require processing mixed inputs of text, images, audio, and video. Attention sparsity in multi-modal scenarios exhibits more complex patterns: interactions between different modalities may require different attention distributions, temporally aligned multi-modal information may need synchronized attention mechanisms, and modality-specific long-range dependencies may require specialized sparse patterns. Future research can explore modality-aware dynamic mask generation, coordination mechanisms for cross-modal attention weights, and specialized sparse pattern designs for different modal characteristics.

Integration with Modern Frameworks. Seamless integration of DMA into mainstream deep learning frameworks, such as PyTorch and the Hugging Face Transformers library, is crucial for its widespread adoption and practical impact. This requires developing a user-friendly and highly optimized implementation that can be easily incorporated into existing model architectures and training pipelines. A key aspect of this integration is the development of efficient, low-level kernels, potentially using Triton or CUDA, to ensure that the performance benefits of DMA are fully realized on modern hardware. Providing a plug-and-play module compatible with the Hugging Face Transformers library would significantly lower the barrier for researchers and practitioners to apply DMA to their models and tasks, thereby fostering further innovation and comparative studies in the field of sparse attention.

### 8 Conclusion

In this paper, we introduced Dynamic Mask Attention, a novel trainable sparse attention mechanism that effectively addresses the key challenges in long-context modeling for large language models. By integrating content-aware dynamic sparse masks with position-aware sparse attention weight computations, Dynamic Mask Attention successfully balances computational efficiency while preserving the ability to retrieve information from long contexts precisely.

Our approach makes several key contributions to the field of efficient attention mechanisms. First, Dynamic Mask Attention achieves computational efficiency comparable to sliding window attention while maintaining the information retrieval capabilities of full attention by retaining a complete, uncompressed key-value cache. Second, by dynamically generating attention masks from value representations, our method enables models to learn which tokens are relevant to the current reasoning process, effectively leveraging both content-aware and position-aware sparsity patterns inherent in language modeling tasks. Third, our specialized hardware-optimized kernel for Dynamic Mask Attention efficiently handles sparse mask regions, translating theoretical computational gains into practical speed improvements.

The comprehensive experimental evaluation demonstrates that Dynamic Mask Attention consistently outperforms existing attention mechanisms across various scales and tasks. In scaling law studies, Dynamic Mask Attention exhibited superior perplexity compared to other attention variants. On challenging tasks like multi-query associative recall, Dynamic Mask Attention demonstrated both effectiveness in information retrieval and computational efficiency. Most significantly, our 1.7B parameter model with Dynamic Mask Attention outperformed the vanilla attention counterpart on standard benchmarks and showed remarkably stronger extrapolation capabilities on the needle-in-a-haystack task when context lengths exceeded the pre-training sequence length.

Dynamic Mask Attention represents a significant step forward in developing efficient and effective attention mechanisms for long-context modeling. By maintaining the full expressive power of attention while reducing computational complexity, our approach enables the development of more capable language models that can effectively process lengthy documents, complex reasoning chains, and rich contextual information. This capability is particularly valuable for applications requiring deep reasoning, code generation, and multi-turn autonomous agents.

Future work could explore adaptive window size selection based on content complexity, create more extrapolation-friendly positional encoding schemes, extend it to multimodal contexts, and develop further theoretical analyses of its properties. We believe that Dynamic Mask Attention provides a promising direction for future research in efficient transformer architectures and will facilitate the development of more powerful and computationally efficient language models.

##### Acknowledgments

We would like to express our gratitude to the OpenSeek project team at Beijing Academy of Artificial Intelligence for their support in developing the hardware kernels. We would also like to thank Professor Albert Gu from Carnegie Mellon University for his valuable guidance and suggestions in connecting the concepts of state-space and self-attention. Additionally, we extend our thanks to all the friends in the community who provided feedback and suggestions, your support has been instrumental in the continuous improvement of this work.

### References

- [1] Martin Arjovsky, Amar Shah, and Yoshua Bengio. “Unitary Evolution Recurrent Neural Networks”. In: The International Conference on Machine Learning (ICML). 2016, pp. 1120–1128.
- [2] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. “Zoology: Measuring and Improving Recall in Efficient Language Models”. In: The International Conference on Learning Representations. 2024.
- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. “Program synthesis with large language models”. In: arXiv preprint arXiv:2108.07732 (2021).
- [4] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. “Longbench: A bilingual, multitask benchmark for long context understanding”. In: arXiv preprint arXiv:2308.14508 (2023).
- [5] Iz Beltagy, Matthew E Peters, and Arman Cohan. “Longformer: The long-document transformer”. In: arXiv preprint arXiv:2004.05150 (2020).
- [6] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. SmolLM-Corpus. July 2024.
- [7] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. “PIQA: Reasoning about Physical Commonsense in Natural Language”. In: Proceedings of the AAAI conference on Artificial Intelligence. Vol. 34. 2020.
- [8] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. “Gpt-NeoX-20B: An Open-source Autoregressive Language Model”. In: arXiv preprint arXiv:2204.06745 (2022).
- [9] Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, et al. “Magicpig: Lsh sampling for efficient llm generation”. In: arXiv preprint arXiv:2410.16179 (2024).
- [10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. “Generating long sequences with sparse transformers”. In: arXiv preprint arXiv:1904.10509 (2019).
- [11] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. “ThinkyouhaveSolvedQuestionAnswering?TryARC,theAI2ReasoningChallenge”.In:arXivpreprintarXiv:1803.05457

(2018).

- [12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. “Training verifiers to solve math word problems”. In: arXiv preprint arXiv:2110.14168 (2021).
- [13] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. “Transformer-xl: Attentive language models beyond a fixed-length context”. In: arXiv preprint arXiv:1901.02860 (2019).
- [14] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. “FLASHATTENTION: fast and memoryefficient exact attention with IO-awareness”. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. 2022.
- [15] Google DeepMind. Gemini 2.5 Pro. Mar. 2025. url: https://blog.google/technology/google- deepmind/ gemini-model-thinking-updates-march-2025.
- [16] Aditya Desai, Shuo Yang, Alejandro Cuadron, Ana Klimovic, Matei Zaharia, Joseph E Gonzalez, and Ion Stoica. “HashAttention: Semantic Sparsity for Faster Inference”. In: arXiv preprint arXiv:2412.14468 (2024).
- [17] Clémentine Fourrier, Nathan Habib, Hynek Kydlíček, Thomas Wolf, and Lewis Tunstall. LightEval: A lightweight framework for LLM evaluation. Version 0.7.0. 2023. url: https://github.com/huggingface/lighteval.
- [18] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A Framework for Few-shot Language Model Evaluation. Version v0.0.1. Sept. 2021. doi: 10.5281/ zenodo.5371628. url: https://doi.org/10.5281/zenodo.5371628.
- [19] Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, et al. “Seerattention: Learning intrinsic sparse attention in your llms”. In: arXiv preprint arXiv:2410.13276 (2024).
- [20] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. “Model tells you what to discard: Adaptive kv cache compression for llms”. In: arXiv preprint arXiv:2310.01801 (2023).

- [21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning”. In: arXiv preprint arXiv:2501.12948 (2025).
- [22] Alex Hägele, Elie Bakouch, Atli Kosson, Leandro Von Werra, Martin Jaggi, et al. “Scaling laws and computeoptimal training beyond fixed training durations”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 76232–76264.
- [23] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. “Measuring massive multitask language understanding”. In: arXiv preprint arXiv:2009.03300 (2020).
- [24] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. “Measuring Massive Multitask Language Understanding”. In: International Conference on Learning Representations. 2021.
- [25] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. “An Empirical Analysis of ComputeOptimal Large Language Model Training”. In: Advances in Neural Information Processing Systems (NeurIPS) 35 (2022), pp. 30016–30030.
- [26] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. “RULER: What’s the Real Context Size of Your Long-Context Language Models?” In: arXiv preprint arXiv:2404.06654 (2024).
- [27] HuggingFace. Open R1: A fully open reproduction of DeepSeek-R1. 2025. url: https://github.com/huggingface/ open-r1.
- [28] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. 2017. arXiv: 1705.03551 [cs.CL].
- [29] GKamradt.LLMTestNeedleInAHaystack.2023.url:https://github.com/gkamradt/LLMTest_NeedleInAHaystack.
- [30] Houyi Li, Wenzhen Zheng, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Shuigeng Zhou, Xiangyu Zhang, and Daxin Jiang. Predictable Scale: Part I – Optimal Hyperparameter Scaling Law in Large Language Model Pretraining. 2025. arXiv: 2503.04715 [cs.LG]. url: https://arxiv.org/abs/2503.04715.
- [31] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. “Snapkv: Llm knows what you are looking for before generation”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 22947–22970.
- [32] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. “Deepseek-v3 technical report”. In: arXiv preprint arXiv:2412.19437 (2024).
- [33] Guangda Liu, Chengwei Li, Jieru Zhao, Chenqi Zhang, and Minyi Guo. “Clusterkv: Manipulating llm kv cache in semantic space for recallable compression”. In: arXiv preprint arXiv:2412.03213 (2024).
- [34] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. “Lost in the middle: How language models use long contexts”. In: Transactions of the Association for Computational Linguistics 12 (2024), pp. 157–173.
- [35] Ilya Loshchilov and Frank Hutter. “Fixing Weight Decay Regularization in Adam”. In: ArXiv abs/1711.05101 (2017). url: https://api.semanticscholar.org/CorpusID:3312944.
- [36] Andre Martins and Ramon Astudillo. “From softmax to sparsemax: A sparse model of attention and multi-label classification”. In: International conference on machine learning. PMLR. 2016, pp. 1614–1623.
- [37] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. “Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering”. In: arXiv preprint arXiv:1809.02789 (2018).
- [38] Meta NVIDIA. PyTorch Container Image. https://catalog.ngc.nvidia.com/orgs/nvidia/containers/ pytorch. 2022.
- [39] CatherineOlsson,NelsonElhage,NeelNanda,NicholasJoseph,NovaDasSarma,TomHenighan,BenMann,Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. “In-context Learning and Induction Heads”. In: Transformer Circuits Thread (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
- [40] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. “The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context”. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. 2016, pp. 1525–1534.

- [41] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. “Generative agents: Interactive simulacra of human behavior”. In: Proceedings of the 36th annual acm symposium on user interface software and technology. 2023, pp. 1–22.
- [42] David W Romero, Anna Kuzina, Erik J Bekkers, Jakub M Tomczak, and Mark Hoogendoorn. “CKConv: Continuous Kernel Convolution For Sequential Data”. In: arXiv preprint arXiv:2102.02611 (2021).
- [43] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. “Winogrande: An Adversarial Winograd Schema Challenge at Scale”. In: Communications of the ACM 64.9 (2021), pp. 99–106.
- [44] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. “Scaling llm test-time compute optimally can be more effective than scaling model parameters”. In: arXiv preprint arXiv:2408.03314 (2024).
- [45] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. “Challenging big-bench tasks and whether chain-of-thought can solve them”. In: Findings of the Association for Computational Linguistics: ACL 2023. 2023, pp. 13003–13051.
- [46] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. “Quest: Query-aware sparsity for efficient long-context llm inference”. In: arXiv preprint arXiv:2406.10774 (2024).
- [47] Qwen Team. Qwen3. Apr. 2025. url: https://qwenlm.github.io/blog/qwen3.
- [48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need”. In: Advances in Neural Information Processing Systems. 2017.
- [49] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. “Transformers: State-of-the-Art Natural Language Processing”. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45.
- [50] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. “Infllm: Training-free long-context extrapolation for llms with an efficient context memory”. In: arXiv preprint arXiv:2402.04617 (2024).
- [51] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. “Effective long-context scaling of foundation models”. In: arXiv preprint arXiv:2309.16039 (2023).
- [52] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. “Native sparse attention: Hardware-aligned and natively trainable sparse attention”. In: arXiv preprint arXiv:2502.11089 (2025).
- [53] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. “Big bird: Transformers for longer sequences”. In: Advances in neural information processing systems 33 (2020), pp. 17283–17297.
- [54] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. “HellaSwag: Can a Machine Really Finish Your Sentence?” In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.
- [55] Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, and Yunhe Feng. DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration. 2025. arXiv: 2506.11104 [cs.CL]. url: https://arxiv.org/abs/ 2506.11104.
- [56] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. “Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges”. In: arXiv preprint arXiv:2401.07339 (2024).
- [57] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. “H2o: Heavy-hitter oracle for efficient generative inference of large language models”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 34661–34710.
- [58] Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang, Rongqiao An, Qi Shi, Zhixing Tan, et al. “LLM MapReduce: Simplified Long-Sequence Processing using Large Language Models”. In: CoRR (2024).

### A Dynamic Mask Attention Implementation

ThefollowinglistingprovidesasampleimplementationoftheDynamicMaskAttentionalgorithminPyTorch, asdescribed in Section 4.

Listing 1: Dynamic Mask Attention implementation in PyTorch

|def dynamic_mask_attention(h_t, position_embeddings, causal_m, past_key_value, W_Q, W_K, W_V, W_dt, A, W_O, num_heads, scaling, keep_window_size): input_shape = h_t.shape[:-1] # [b, q_len] hidden_shape = (*input_shape, -1, h_t.shape[-1] // num_heads) # linear projections q_t = W_Q(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h] k_t = W_K(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h] v_t = W_V(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h]<br><br>o_t = torch.zeros_like(q_t) # [b, n_h, q_len, d_h] # apply rotary position embeddings q_t, k_t = apply_rotary_pos_emb(q_t, k_t, *position_embeddings) # concatenate past key and value states k, v = past_key_value.update(k_t, v_t) # [b, n_h, k_len, d_h] # calculate dynamic mask dt = W_dt(v.transpose(1, 2).reshape(v.shape[0], v.shape[-2], -1)) # [b, k_len, n_h] dt = torch.exp(A * F.softplus(dt)).transpose(-1, -2) # [b, n_h, k_len] m_t = dt[:, :, None, :].expand(-1, -1, h_t.shape[1], -1) # [b, n_h, q_len, k_len] active_m = torch.zeros_like(m_t) m_t = m_t.masked_fill(causal_m != 0, -float('inf')) topk_indices = torch.topk(m_t, keep_window_size, dim=-1, sorted=False).indices active_m = active_m.scatter(-1, topk_indices, 1.0) m_t = m_t.masked_fill(active_m == 0.0, -float('inf')) # calculate sparse attention weight for b_idx in range(hidden_shape[0]): # b<br><br>for h_idx in range(num_heads): # n_h<br><br>for q_idx in range(hidden_shape[1]): # q_len q_elem = q_t[b_idx, h_idx, q_idx, :] # [d_h] indices = topk_indices[b_idx, h_idx, q_idx] # [w] k_vecs = k[b_idx, h_idx, indices, :] # [w, d_h] v_vecs = v[b_idx, h_idx, indices, :] # [w, d_h] a_elem = torch.sum(q_elem.unsqueeze(0) * k_vecs, dim=-1) # [w] a_elem = a_elem * scaling + m_t[b_idx, h_idx, q_idx, indices] a_elem = F.softmax(a_elem, dim=-1)<br><br>o_elem = torch.sum(a_elem.unsqueeze(1) * v_vecs, dim=0) # [d_h]<br><br>o_t[b_idx, h_idx, q_idx, :] = o_elem<br><br><br>o_t = o_t.transpose(1, 2).contiguous() # [b, q_len, n_h, d_h] h_t = W_O(o_t.view(*input_shape, -1)) # [b, q_len, d_model] return h_t<br><br><br>|
|---|

The implementation demonstrates the core computational flow of the Dynamic Mask Attention mechanism. First, the query, key, and value matrices are computed through linear projections, followed by the application of rotary position embeddings. The core innovation of the algorithm is then reflected in the dynamic mask generation process: dynamic weights 𝛿 are calculated from the value vectors, and a sparse mask is generated using the topk operation, retaining only the most relevant 𝑤 key-value pairs. Finally, in the sparse attention computation phase, the algorithm computes attention weights only for the selected key-value pairs, significantly reducing computational complexity. In actual kernel implementations, it is possible to check if there are any active tokens in the MMA block; if not, the computation for that block can be skipped.

### B Experiment Setup

To make the comparison between attention variants fair and reproducible, we standardize the model, data pipeline, optimization, and evaluation, only changing the attention module and its related hyperparameters. All experiments were conducted using the open-source PyTorch images (NVIDIA 2022) and the Transformers framework (Wolf et al. 2020). We use SmolLMCorpus (Ben Allal et al. 2024) as training data. For evaluation frameworks, we utilized the LM evaluation harness (L. Gao et al. 2021) from EleutherAI for perplexity tasks, and the lighteval (Fourrier et al. 2023) from HuggingFace for downstream tasks. Table 4 lists the model sizes and key hyperparameters used at each scale. We keep the depth, width, and number of heads consistent across variants under the same parameter budget, only changing the specific parameters of the attention variants, with parameter symbols consistent with those in the original papers of the attention variants.

We summarize the meaning of the columns in Table 4 and clarify which hyperparameters are used by each attention variant.

- • Params: total number of model parameters.
- • Steps: training steps.
- • Batch: tokens per step.
- • LR: peak learning rate.
- • 𝑛𝑙𝑎𝑦𝑒𝑟𝑠: number of Transformer layers.
- • 𝑑𝑚𝑜𝑑𝑒𝑙: model hidden size.
- • 𝑛ℎ: number of attention heads.
- • 𝑛ℎ𝑘𝑣: number of KV heads. In all our configurations we set 𝑛ℎ𝑘𝑣 = 𝑛ℎ/2.
- • MHA (full attention): standard scaled dot-product attention. Sparse-specific columns (𝑤, 𝑑𝑐, 𝐵, 𝐵′, 𝑘) are not used.
- • SWA (sliding-window attention): 𝑤 is the sliding window size; other sparse-specific columns are not used.
- • MLA (multi-head latent attention): 𝑑𝑐 is the latent/compression dimension; other columns are not used.
- • NSA (native sparse attention): 𝑑𝑐 is the compression dimension, 𝑤 is the sliding window size, 𝐵 is the compressing block size, 𝐵′ is the selection block size, and 𝑘 is the num selected blocks. All settings following the original paper.
- • DMA (dynamic mask attention): 𝑤 is the per-head top-𝑤 kept keys for dynamic masks; other sparse-specific columns are not used.

- 1The implementation code for MHA is available at https://github.com/Dao-AILab/flash-attention.
- 2The implementation code for SWA is available at https://github.com/Dao-AILab/flash-attention.
- 3The implementation code for MLA is available at https://github.com/deepseek-ai/FlashMLA.
- 4The implementation code for NSA is available at https://github.com/lucidrains/native-sparse-attention-pytorch.

- Table 4: Self-Attention Variants Scaling Laws Configurations. The model and hyperparameter configurations used in our self-attention variants scaling laws experiments.

Algos Params Steps Batch LR 𝑛𝑙𝑎𝑦𝑒𝑟𝑠 𝑑𝑚𝑜𝑑𝑒𝑙 𝑛ℎ 𝑛ℎ𝑘𝑣 𝑤 𝑑𝑐 𝐵 𝐵′ 𝑘 MHA ≈ 80M 13,500 0.128M tokens 3e-3 12 768 6 3 - - - - SWA ≈ 80M 13,500 0.128M tokens 3e-3 12 768 6 3 1024 - - - MLA ≈ 80M 13,500 0.128M tokens 3e-3 12 768 6 3 - 192 - - NSA ≈ 80M 13,500 0.128M tokens 3e-3 12 768 6 3 512 192 32 64 16 DMA ≈ 80M 13,500 0.128M tokens 3e-3 12 768 6 3 1024 - - - MHA ≈ 200M 20,800 0.192M tokens 2e-3 16 1024 8 4 - - - - SWA ≈ 200M 20,800 0.192M tokens 2e-3 16 1024 8 4 1024 - - - MLA ≈ 200M 20,800 0.192M tokens 2e-3 16 1024 8 4 - 256 - - NSA ≈ 200M 20,800 0.192M tokens 2e-3 16 1024 8 4 512 192 32 64 16 DMA ≈ 200M 20,800 0.192M tokens 2e-3 16 1024 8 4 1024 - - - MHA ≈ 680M 35,000 0.392M tokens 1e-3 24 1536 12 6 - - - - NSA ≈ 680M 35,000 0.392M tokens 1e-3 24 1536 12 6 512 192 32 64 16 DMA ≈ 680M 35,000 0.392M tokens 1e-3 24 1536 12 6 1024 - - - MHA ≈ 1.7B 40,000 1M tokens 1e-3 28 2048 16 8 - - - - NSA ≈ 1.7B 40,000 1M tokens 1e-3 28 2048 16 8 512 256 32 64 16 DMA ≈ 1.7B 40,000 1M tokens 1e-3 28 2048 16 8 2048 - - - -

- Table 5: Speed Benchmark Configurations. The common settings for running time curves and the sparse hyperparameters for each method.

Algo warmups runs 𝑛ℎ 𝑛ℎ𝑘𝑣 𝑑ℎ 𝑤 𝑑𝑐 𝐵 𝐵′ 𝑘 precision

MHA1 3 1,000 32 8 128 - - - - - bf16 SWA2 3 1,000 32 8 128 1024 - - - - bf16 MLA3 3 1,000 32 8 128 - 192 - - - bf16 NSA4 3 1,000 32 8 128 512 256 32 64 16 bf16 DMA 3 1,000 32 8 128 1024 - - - - bf16

### C Attention Heatmaps

Dynamic Mask Attention Heatmaps

Head 0

Head 1

Head 2

Head 3

01020304050

01020304050

01020304050

01020304050

QueryPosition

QueryPosition

QueryPosition

QueryPosition

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

Key Position

Key Position

Key Position

Key Position

Head 4

Head 5

Head 6

Head 7

01020304050

01020304050

01020304050

01020304050

QueryPosition

QueryPosition

QueryPosition

QueryPosition

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

Key Position

Key Position

Key Position

Key Position

Head 8

Head 9

Head 10

Head 11

01020304050

01020304050

01020304050

01020304050

QueryPosition

QueryPosition

QueryPosition

QueryPosition

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

Key Position

Key Position

Key Position

Key Position

Head 12

Head 13

Head 14

Head 15

01020304050

01020304050

01020304050

01020304050

QueryPosition

QueryPosition

QueryPosition

QueryPosition

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

0 10 20 30 40 50

Key Position

Key Position

Key Position

Key Position

- Figure 11: Full Heatmaps of Dynamic Mask Attention. The heatmaps show the attention weights of each head in the Dynamic Mask Attention mechanism, indicating which tokens each head focuses on.

