# arXiv:2509.00605v1[cs.CL]30Aug2025

## Gated Associative Memory: A Parallel O(N) Architecture for Efficient Sequence Modeling

Rishiraj Acharya1

1Independent Researcher heyrishiraj@gmail.com

Abstract

The Transformer architecture, underpinned by the self-attention mechanism, has become the de facto standard for sequence modeling tasks. However, its core computational primitive scales quadratically with sequence length (O(N2)), creating a significant bottleneck for processing long contexts. In this paper, we propose the Gated Associative Memory (GAM) network, a novel, fully parallel architecture for sequence modeling that exhibits linear complexity (O(N)) with respect to sequence length. The GAM block replaces the self-attention layer with two parallel pathways: a causal convolution to efficiently capture local, position-dependent context, and a parallel associative memory retrieval mechanism to model global, content-based patterns. These pathways are dynamically fused using a gating mechanism, allowing the model to flexibly combine local and global information for each token. We implement GAM from scratch and conduct a rigorous comparative analysis against a standard Transformer model and a modern linear-time baseline (Mamba) on the WikiText-2 benchmark, as well as against the Transformer on the TinyStories dataset. Our experiments demonstrate that GAM is consistently faster, outperforming both baselines on training speed, and achieves a superior or competitive final validation perplexity across all datasets, establishing it as a promising and efficient alternative for sequence modeling.

### 1 Introduction

Since its introduction, the Transformer [11] has revolutionized the field of natural language processing. Its central innovation, the self-attention mechanism, allows for rich, pairwise interactions between all tokens in a sequence, capturing complex dependencies regardless of their distance. This capability has led to state-of-the-art results across a vast array of tasks.

However, this expressive power comes at a steep computational cost. The self-attention mechanism requires a dot-product between Query and Key matrices of size (N, d), resulting in an attention map of size (N, N), where N is the sequence length and d is the model dimension. This leads to computational and memory complexity of O(N2d), which is prohibitive for applications involving very long sequences, such as high-resolution document summarization, genomic data analysis, or processing lengthy video streams.

This quadratic bottleneck has spurred a wave of research into ”Efficient Transformers” [10], which aim to approximate the self-attention matrix using methods like sparsity [1], low-rank factorization [12], or kernelization [2]. While successful, these methods often introduce architectural complexity or trade-offs in expressivity. Another line of work has revisited recurrent neural networks (RNNs) (e.g., LSTMs, GRUs), which are naturally O(N) [6]. However, their inherently sequential nature makes them difficult to parallelize on modern hardware, often leading to slower training times despite their theoretical efficiency. More recent architectures like State Space Models (SSMs) have shown great promise in achieving linear-time performance.

Models like Mamba [4], in particular, have demonstrated strong performance by using a selection mechanism and a hardware-aware parallel scan algorithm. While these models are highly effective, they reintroduce a form of recurrence into their design.

In this work, we address these challenges by introducing the Gated Associative Memory (GAM) network. GAM is designed from the ground up to satisfy two critical criteria: (1) linear computational complexity and (2) maximum parallelizability on modern accelerators by avoiding recurrence entirely. It replaces the self-attention block with a novel GAMBlock that combines two complementary context-modeling pathways:

- 1. A Local Context Pathway: A 1D causal convolution efficiently captures local syntactic and positional relationships.
- 2. A Global Context Pathway: A parallel associative memory retrieves global, contentbased patterns from a learned memory bank for all tokens simultaneously.

These pathways are fused with a learnable gating mechanism, enabling the network to dynamically allocate resources to local or global context as needed. Our contributions are:

- • We propose the Gated Associative Memory (GAM) architecture, a novel O(N) sequence model that is fully parallelizable and non-recurrent.
- • We provide a complete implementation and empirically demonstrate that GAM consistently trains faster than both a comparable Transformer and Mamba on a standard GPU.
- • We show through experiments on the WikiText-2 and TinyStories datasets that GAM achieves better perplexity than a well-trained Transformer baseline and the Mamba baseline, highlighting its effectiveness and generalizability.

### 2 Related Work

The quest for efficient sequence modeling beyond the standard Transformer has been a vibrant area of research. Our work is situated within this landscape and draws inspiration from several key ideas.

Efficient Transformers A large body of work, surveyed by Tay et al. [10], focuses on approximating the dense N × N attention matrix. These methods can be broadly categorized:

- • Sparsity-based Methods: Models like Longformer [1] use a combination of local windowed attention and sparse global attention to reduce computation, allowing them to process thousands of tokens.
- • Low-Rank Methods: Linformer [12] is based on the observation that the self-attention matrix is often low-rank and can be approximated by projecting the Key and Value matrices to a lower-dimensional space, reducing complexity from O(N2) to O(N).
- • Kernel-based Methods: Performers [2] use random feature maps to approximate the softmax kernel, enabling a linear-time attention mechanism without direct computation of the N × N matrix.

While effective, these approaches primarily modify the self-attention mechanism itself. GAM, in contrast, replaces it entirely with a different inductive bias.

Recurrent Models and State Space Models (SSMs) Before Transformers, RNNs, and particularly LSTMs [6], were the dominant paradigm for sequence modeling. Their O(N) complexity is a key advantage, but their sequential nature limits training parallelism. Recently, there has been a resurgence of interest in models that combine recurrence with modern hardwareaware designs. Structured State Space Models (S4) [5] and Mamba [4] are prominent examples that achieve linear-time scaling and strong performance by drawing on principles from classical state-space theory. Mamba, in particular, introduces a selective SSM that allows the model to dynamically focus on or ignore inputs based on context, implemented via a highly optimized, hardware-aware scan operation. While Mamba achieves impressive performance through this recurrent scan, GAM pursues a different path to efficiency by avoiding recurrence altogether, relying instead on fully parallel primitives (convolution and matrix multiplication) that are inherently well-suited to modern accelerators.

Convolutional Sequence Models Using convolutions for sequence tasks is not new. Models like TCNs (Temporal Convolutional Networks) have shown that causal convolutions can be very effective at capturing historical context. GAM incorporates this idea in its local pathway, using it as a specialized and efficient mechanism for positional and syntactic information.

GAM’s novelty lies in its explicit and parallel decomposition of context into local (convolutional) and global (associative memory) streams, which are then dynamically combined with a learned gate. This hybrid approach avoids the approximations of many efficient Transformers and the sequential bottlenecks of RNNs, offering a distinct and effective design point in the space of efficient sequence architectures.

### 3 The Gated Associative Memory (GAM) Architecture

The GAM model is a stack of L identical blocks, similar to a Transformer. It processes an input sequence of token embeddings and produces a sequence of contextualized output vectors. The core innovation lies within the GAMBlock, which replaces the Multi-Head Self-Attention sublayer. The overall model follows a standard structure with token and positional embeddings, a stack of GAM blocks, and a final layer normalization and language model head.

#### 3.1 The GAM Block

The GAM block is designed for maximal parallelism and linear-time computation. As shown in Figure 1, an input x is first normalized. It then branches into parallel pathways to compute local and global context, which are fused via a learned gating mechanism. This is followed by a residual connection and a standard feed-forward network (FFN) sub-layer.

#### 3.2 Local Context Pathway: Causal Convolution

To capture local structure, such as n-gram relationships and word order, we employ a 1D causal convolution. A convolution with kernel size k allows a token to gather information from its k − 1 predecessors. To ensure causality (i.e., information only flows from past to future), we apply asymmetric padding of k − 1 to the beginning of the sequence. For an input sequence X of shape (B, N, d), where B is batch size, N is sequence length, and d is the model dimension, the operation is:

- 1. Permute: The input is reshaped to (B, d, N) to match the Conv1d expectation.
- 2. Pad: Asymmetric padding of (k − 1) is applied to the time dimension.
- 3. Convolve: The convolution is applied.

[Figure 1] Architecture for Efficient Sequence Modeling_images/imageFile1.png>)

- Figure 1: The GAM Block. The input x first passes through a Layer Normalization. It then splits into three branches. The first branch is a residual connection. The second branch computes local and global context in parallel. The local context is generated by a Causal 1D Convolution. The global context is generated by querying a learnable Memory Bank. The outputs of these two pathways are combined via a learned gate. The gated output is added to the residual connection. This is followed by another Layer Normalization and a standard Feed-Forward Network, also with a residual connection.

- 4. Trim & Permute: The output is trimmed to the original length N and permuted back to (B, N, d).

This operation is highly parallelizable and has a complexity of O(N · k · d), which is linear in sequence length N. We use a depthwise convolution (groups=d), where each feature channel is processed independently, to further improve efficiency and reduce parameters.

#### 3.3 Global Context Pathway: Parallel Associative Memory

This pathway is designed to model long-range, content-based dependencies, replacing the quadratic self-attention mechanism. It consists of a learnable Memory Bank M, a matrix of size (num slots, d model), initialized with Xavier uniform initialization. Each row of M can be interpreted as a learnable ”prototypical” contextual pattern that is learned from the data.

For an input sequence X of shape (B, N, d), the retrieval process is performed for all N tokens in parallel:

- 1. Similarity Scoring: The model computes the dot-product similarity between each of the N token representations and every one of the num slots in the memory bank. This is a single matrix multiplication:

Scores = XMT (1) The resulting Scores tensor has the shape (B, N, num slots).

- 2. Soft Retrieval: A softmax function is applied over the num slots dimension for each token. This yields a set of attention-like weights, indicating which prototypical patterns in M are most relevant for each token.

Weights = softmax(Scores) (2) The Weights tensor has the shape (B, N, num slots).

- 3. Context Aggregation: The final global context is a weighted average of the memory bank vectors, computed via a single matrix multiplication between the retrieval weights and the memory bank itself:

GlobalContext = Weights · M (3) The resulting GlobalContext tensor has the shape (B, N, d).

This entire process involves only matrix multiplications with fixed-size matrices (M), making its complexity O(N · num slots · d). Since num slots and d are fixed hyperparameters, the complexity is linear with respect to the sequence length N.

#### 3.4 Gating and Fusion

The local and global context pathways offer complementary views of the sequence. The causal convolution is an expert on local syntax and strict ordering, while the associative memory is an expert on high-level, content-based patterns that are position-agnostic. To combine them effectively, we use a dynamic gating mechanism.

For each token’s input representation x, a gate vector is computed using a single linear

layer, which is then split into two halves, glocal and gglobal. These gates modulate the flow of information from each pathway:

g = Linear(x) (4) glocal,gglobal = chunk(g,2,dim = −1) (5) FusedContext = σ(glocal) · LocalContext + σ(gglobal) · GlobalContext (6)

The sigmoid function σ squashes the gate values to the range (0, 1), acting as a soft switch. This allows the model to learn, on a per-token basis, whether to prioritize local syntactic cues (e.g., for function words) or global semantic information (e.g., for content words). This FusedContext is then added to the original input via the residual connection and passed to the FFN.

### 4 Experimental Setup

We conducted experiments to compare our proposed GAM model against a standard Transformer baseline and a Mamba baseline on causal language modeling tasks.

#### 4.1 Datasets and Preprocessing

- • WikiText-2: We used the wikitext-2-raw-v1 version of the dataset [8], a standard benchmark for language modeling consisting of high-quality Wikipedia articles.
- • TinyStories: We also used the roneneldan/TinyStories dataset [3], a large corpus of short stories generated by GPT-3.5/4 that use a vocabulary typical of a 3-4 year old. This dataset is designed to test a model’s ability to learn fundamental language structures and reasoning in a constrained setting.

For each dataset, we trained a Byte-Pair Encoding (BPE) tokenizer from scratch on its respective training corpus. The final vocabulary size was set to 10,000 tokens. All documents were concatenated and then split into fixed-length chunks of 256 tokens to form the input sequences.

- 4.2 Models To ensure a fair comparison, all models were built with similar scale and hyperparameters.

- • GAM (ours): A 6-layer GAM model with an embedding dimension dmodel=512. The associative memory contained num slots=512, and the causal convolution used a kernel size of k = 3. A dropout rate of 0.1 was applied. This resulted in a total of 22.6 million trainable parameters.

- • Transformer (baseline): A 6-layer GPT-style decoder-only Transformer with dmodel=512 and nhead=8, resulting in a head dimension of 64. A dropout rate of 0.1 was used throughout the model. This resulted in a total of 24.2 million trainable parameters.

- • Mamba (baseline): A 6-layer Mamba model with dmodel=512. We used the official implementation with standard parameters: state space dimension dstate=16, convolution kernel dconv=4, and expansion factor ‘expand‘=3. A dropout rate of 0.1 was applied. This resulted in a total of 20.5 million trainable parameters. Mamba was evaluated on the WikiText-2 benchmark to compare GAM against another prominent O(N) architecture.

#### 4.3 Training Details

All models were trained from scratch for 5 epochs on each dataset using a single NVIDIA T4 GPU. We used the AdamW optimizer [7] with a learning rate of 3 × 10−4, betas of (0.9, 0.95), and a weight decay of 0.1. A learning rate scheduler with a linear warmup of 100 steps followed by a cosine decay was used for all models to ensure stable training. The batch size was set to 32. Gradient clipping was applied at a norm of 1.0.

##### 4.4 Evaluation Metrics We evaluated the models on two primary axes:

- 1. Accuracy: Measured by Perplexity (PPL) on the validation set, calculated as exp(cross entropy loss). Lower perplexity indicates a better language model.

- 2. Efficiency: Measured by the average wall-clock time per training epoch in seconds.

### 5 Results and Analysis

The results of our comparative experiments are summarized in Table 1. Across two diverse datasets, the GAM architecture consistently demonstrates advantages in both training speed and modeling performance, outperforming both the quadratic Transformer and the linear Mamba baseline on WikiText-2.

- Table 1: Comparison of GAM, Transformer, and Mamba. GAM is the fastest and achieves the lowest (better) perplexity on WikiText-2. It also outperforms the Transformer on TinyStories.

Avg. Time / Epoch (s)

Dataset Model Params

Val. Loss Val. PPL WikiText-2 Transformer 24.2 M 131.9 s 6.8233 918.99

Mamba 20.5 M 127.1 s 6.9251 1017.54 GAM (Ours) 22.6 M 117.2 s 6.7828 882.57

TinyStories Transformer 24.2 M 671.6 s 3.1591 23.55 GAM (Ours) 22.6 M 601.4 s 3.1418 23.15

- 5.1 Efficiency Analysis The GAM model demonstrated a significant and consistent efficiency advantage.

- • On WikiText-2, GAM’s average epoch time of 117.2s was the fastest, outperforming Mamba (127.1s) by 7.8% and the Transformer (131.9s) by 11.1%. This highlights the efficiency of GAM’s fully parallel, non-recurrent design, which avoids the scan operations inherent to SSMs, even when those operations are highly optimized.
- • On TinyStories, a much larger dataset, GAM maintained a 10.5% speed advantage over the Transformer (601.4s vs. 671.6s per epoch).

These results empirically validate our architectural design goals. By replacing the O(N2) selfattention with fully parallelizable O(N) operations (causal convolution and associative memory retrieval), GAM better utilizes the parallel processing capabilities of modern GPUs. This leads to a direct and consistent improvement in training throughput, even at a moderate sequence length of 256.

#### 5.2 Scaling Benchmark

While the full training runs demonstrate GAM’s efficiency at a fixed sequence length of 256, the theoretical O(N) advantage becomes most apparent as the sequence length N increases. To empirically validate this, we conducted a targeted scaling benchmark that isolates the computational and memory costs of a single GAM block versus a single Transformer block.

In this benchmark, we measured the average forward-and-backward pass time and the peak allocated GPU memory for both blocks. We used a fixed batch size of 16 and an embedding dimension of 512, while systematically increasing the sequence length from 256 to 8192. The results, shown in Table 2 and visualized in Figure 2, unequivocally demonstrate the practical implications of their differing complexities.

- Table 2: Scaling benchmark results for a single block. Time is the average for a forward and backward pass. Memory is the peak GPU memory allocated. The Transformer fails due to an Out-of-Memory (OOM) error at longer sequences, while GAM scales linearly.

Time (ms) Peak Memory (MB) Sequence Length GAM Transformer GAM Transformer 256 8.97 8.90 179.42 216.03 512 13.09 23.86 325.48 552.98 1024 25.86 74.19 617.60 1964.79 2048 51.94 279.37 1201.85 7483.92 4096 105.03 Failed (OOM) 2370.35 Failed (OOM) 8192 217.30 Failed (OOM) 4707.35 Failed (OOM)

[Figure 2] Architecture for Efficient Sequence Modeling_images/imageFile2.png>)

- Figure 2: GAM vs. Transformer Scaling Comparison. (Left) Average forward-backward pass time vs. sequence length. (Right) Peak GPU memory usage vs. sequence length. Both axes are on a logarithmic scale. The Transformer’s quadratic growth is evident in the steep upward curve, while GAM exhibits clear linear scaling. The Transformer fails due to out-ofmemory errors beyond a sequence length of 2048 in this setup.

At a short sequence length of 256, the models have nearly identical runtimes, as fixed-

cost operations (like the feed-forward network) dominate. However, the performance rapidly diverges. As the sequence length doubles, the Transformer’s runtime and memory usage roughly quadruple, a clear sign of its O(N2) complexity. In stark contrast, GAM’s resource consumption approximately doubles, consistent with its O(N) design.

- 5.3 Performance Analysis In terms of accuracy, the GAM model achieved superior performance on both datasets.

- • On WikiText-2, GAM achieved a final validation perplexity of 882.57, clearly outperforming both the Transformer baseline (918.99) and the Mamba baseline (1017.54). This shows that our simplified, parallel architecture does not sacrifice modeling capability, and can in fact be more effective than other strong baselines.
- • On TinyStories, GAM also achieved a better final perplexity of 23.15 compared to the Transformer’s 23.55. This demonstrates that GAM’s architectural priors are also effective on text with simpler syntax and a focus on narrative coherence.

The validation learning curves, shown in Figures 3 and 4, illustrate that GAM not only reaches a better final score but often maintains a performance advantage throughout the training process. This strong performance across two textually different domains suggests that the explicit decomposition of context modeling is a robust strategy. The causal convolution provides a stable, hard-coded mechanism for local syntax, while the associative memory can focus entirely on learning and retrieving high-level semantic patterns. This clear division of labor proves to be highly effective.

[Figure 3] Architecture for Efficient Sequence Modeling_images/imageFile3.png>)

- Figure 3: Training dynamics on the WikiText-2 dataset. (a) Validation perplexity, (b) validation loss, and (c) wall-clock time per epoch. The plots show GAM (in blue) achieving a lower final perplexity and consistently faster epoch times compared to both the Transformer (in red) and Mamba (in green) baselines.

[Figure 4] Architecture for Efficient Sequence Modeling_images/imageFile4.png>)

- Figure 4: Training dynamics on the TinyStories dataset. (a) Validation perplexity, (b) validation loss, and (c) wall-clock time per epoch. GAM (in green) demonstrates a faster learning trajectory and maintains a significant efficiency advantage throughout the 5 epochs compared to the Transformer (in violet).

#### 5.4 Ablation Study

To dissect the GAM architecture and validate the contribution of its core components, we conducted a series of ablation studies on the WikiText-2 dataset. We evaluated four configurations of the GAM model:

- 1. GAM (Full): The complete proposed model with both local (convolution) and global (associative memory) pathways, fused by the learned gating mechanism.
- 2. GAM (Sum Fusion): A model with both pathways, but with the gating mechanism removed. The outputs are combined via simple element-wise addition.
- 3. GAM (Global Only): A model that uses only the parallel associative memory pathway, removing the causal convolution entirely.
- 4. GAM (Local Only): A model that uses only the causal convolution pathway, removing the associative memory. This is analogous to a Temporal Convolutional Network (TCN).

The results, summarized in Table 3, clearly demonstrate the distinct roles and the combined power of each component.

- Table 3: Ablation study of the GAM architecture on WikiText-2. All components—the local pathway, the global pathway, and particularly the gating mechanism—are shown to be essential for achieving the best performance. PPL is the best validation perplexity achieved during training.

Model Configuration Gating? Local? Global? Params Val. PPL GAM (Full) ✓ ✓ ✓ 22.6 M 900.84

GAM (Global Only) × × ✓ 19.4 M 905.45 GAM (Sum Fusion) × ✓ ✓ 19.4 M 942.59 GAM (Local Only) × ✓ × 17.9 M 944.70

The full GAM model achieves the lowest perplexity, confirming the effectiveness of the complete architecture. The analysis of the ablated models provides several key insights:

First, the gating mechanism is crucial for effective fusion. By simply removing the gate and using summation (GAM (Sum Fusion)), the perplexity degrades significantly from 900.84 to 942.59. This indicates that a static, unweighted combination of local and global contexts is suboptimal. The learned gate provides a necessary, dynamic control mechanism that allows the model to intelligently allocate resources between the two information streams on a per-token basis.

Second, the global associative memory is the primary driver of performance. The GAM (Global Only) model performs remarkably well, achieving a perplexity of 905.45, very close to the full model. This suggests that for a general language modeling task like WikiText-2, capturing long-range, content-based semantic patterns is the most critical factor. However, the fact that the full model still performs better demonstrates that the local context provided by the convolution, while not sufficient on its own, adds indispensable and complementary information about word order and syntax.

Finally, local context alone is insufficient. The GAM (Local Only) model performs the worst, with a perplexity of 944.70. This confirms that while convolutions can effectively model local structure, they fail to capture the long-distance dependencies required for robust language understanding.

In conclusion, this ablation study validates our core architectural hypothesis. The GAM model’s strength lies not in any single component, but in the synergistic combination of a

powerful global memory pathway and a refined local syntax pathway, which are effectively integrated by a dynamic gating mechanism.

#### 5.5 Discussion

The results strongly suggest that the quadratic complexity of self-attention is not strictly necessary to achieve high performance in language modeling. Our GAM architecture provides a compelling alternative that is both computationally cheaper and more performant on the benchmarks tested. The consistency of these results across a complex, factual dataset (WikiText-2) and a simple, narrative dataset (TinyStories) highlights the generalizability of GAM’s design. The fact that it outperforms not only the Transformer but also the efficient Mamba baseline on WikiText-2 underscores the effectiveness of its fully parallel, non-recurrent inductive biases.

### 6 Conclusion and Future Work

In this paper, we introduced the Gated Associative Memory (GAM) network, a novel O(N) architecture for sequence modeling. By combining a causal convolution for local context with a parallel associative memory for global patterns, GAM achieves significant and consistent gains in computational efficiency while simultaneously improving modeling accuracy over both a standard Transformer baseline and a strong linear-time competitor, Mamba. Our experiments on WikiText-2 and TinyStories confirm that GAM is faster and obtains better perplexity scores.

This work opens several avenues for future research. The most immediate next step is to evaluate GAM on tasks with much longer sequences, such as those found in the Long Range Arena benchmark [9], where its linear complexity advantage should become even more pronounced. Furthermore, scaling up the model size and training on larger datasets will be crucial to determine its performance ceiling against state-of-the-art models. Finally, the learned memory bank in the GAM model is a rich source of information; analyzing the contextual patterns captured by the memory slots could provide valuable insights into how language models represent knowledge.

### References

- [1] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer, 2020. URL https://arxiv.org/abs/2004.05150.
- [2] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. Rethinking attention with performers, 2022. URL https://arxiv.org/abs/2009.14794.
- [3] Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english?, 2023. URL https://arxiv.org/abs/2305.07759.
- [4] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces,

2024. URL https://arxiv.org/abs/2312.00752.

- [5] Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces, 2022. URL https://arxiv.org/abs/2111.00396.
- [6] Sepp Hochreiter and Ju¨rgen Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.
- [7] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/1711.05101.
- [8] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. URL https://arxiv.org/abs/1609.07843.
- [9] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers, 2020. URL https://arxiv.org/abs/2011.04006.
- [10] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey, 2022. URL https://arxiv.org/abs/2009.06732.
- [11] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. URL https: //arxiv.org/abs/1706.03762.
- [12] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Selfattention with linear complexity, 2020. URL https://arxiv.org/abs/2006.04768.

